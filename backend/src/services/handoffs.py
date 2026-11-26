"""Persist instrument-conversion handoffs and their immutable review snapshots."""

import json
import math

from fastapi import HTTPException
from sqlalchemy import select

from src.constants.workflow import Decision, HandoffStatus, Role, RunStatus
from src.domain.project import SPEC, analyze_conversion, merge_pair_rows
from src.models.handoff import Handoff, HandoffReview
from src.models.paired_dataset import PairedDataset
from src.models.run import Run
from src.services.audit import record
from src.services.queue import notify_worker
from src.utils.time import utc_stamp

CONVERSION_PARAMETERS = {item['key']: item for item in SPEC['conversion_settings']}


def validate_conversion_settings(settings):
    if settings.keys() - CONVERSION_PARAMETERS.keys():
        raise ValueError('存在本项目未定义的换算参数')
    checked = {}
    for key, parameter in CONVERSION_PARAMETERS.items():
        value = settings.get(key, parameter['default'])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{parameter['label']}须为数值")
        if not parameter['min'] <= value <= parameter['max']:
            raise ValueError(f"{parameter['label']}超出范围")
        if parameter['step'] == 1 and value != int(value):
            raise ValueError(f"{parameter['label']}必须是整数")
        checked[key] = int(value) if parameter['step'] == 1 else value
    return checked


def get_paired_dataset(session, dataset_id):
    dataset = session.get(PairedDataset, dataset_id)
    if dataset is None:
        raise LookupError('找不到对照台账')
    return dataset


def handoff_info(handoff, review):
    return {
        'id': handoff.id, 'label': handoff.label, 'base_run_id': handoff.base_run_id,
        'dataset_ids': list(handoff.dataset_ids or []), 'previous_handoff_id': handoff.previous_handoff_id,
        'anchor_instrument': handoff.anchor_instrument, 'created_at': handoff.created_at,
        'status': handoff.status, 'settings': handoff.settings,
        'decision': review.decision if review else None,
        'ready': handoff.result['handoff']['ready'] if handoff.result else None,
        'superseded_by': handoff.superseded_by,
    }


def list_handoffs(session, actor):
    statement = (select(Handoff, HandoffReview)
                 .outerjoin(HandoffReview, Handoff.id == HandoffReview.handoff_id)
                 .order_by(Handoff.id.desc()).limit(100))
    if actor.role == Role.VIEWER:
        statement = statement.where(HandoffReview.decision == Decision.APPROVED)
    return [handoff_info(handoff, review) for handoff, review in session.execute(statement)]


def get_handoff(session, handoff_id, actor):
    handoff = session.get(Handoff, handoff_id)
    review = session.get(HandoffReview, handoff_id)
    if handoff is None or (actor.role == Role.VIEWER and (not review or review.decision != Decision.APPROVED)):
        raise LookupError('找不到可查看的换算交接')
    return handoff, review


def create_handoff(session, payload, actor):
    base = session.get(Run, payload.base_run_id)
    if base is None:
        raise LookupError('找不到原模型运行')
    if base.status != RunStatus.COMPLETED or base.result is None:
        raise ValueError('原模型运行尚未成功完成，无法据此建立换算交接。')
    datasets = [get_paired_dataset(session, dataset_id) for dataset_id in payload.dataset_ids]
    previous = None
    if payload.previous_handoff_id is not None:
        previous = session.get(Handoff, payload.previous_handoff_id)
        if previous is None:
            raise LookupError('找不到被复核的原交接')
        if previous.result is None or previous.status != HandoffStatus.COMPLETED:
            raise ValueError('原交接尚未成功完成，无法复核。')
        previous_review = session.get(HandoffReview, previous.id)
        if previous_review is None or previous_review.decision != Decision.APPROVED:
            raise ValueError('仅已发出的交接可纳入晚到对照数据复核。')
        if previous.base_run_id != base.id:
            raise ValueError('复核须以原交接所用的同一冻结模型为基准。')
        if previous.superseded_by is not None:
            raise HTTPException(409, '该交接已被复核替换，不能再次复核')
    handoff = Handoff(
        label=payload.label, base_run_id=base.id,
        dataset_ids=[dataset.id for dataset in datasets],
        previous_handoff_id=previous.id if previous else None,
        anchor_instrument=payload.anchor_instrument, created_by=actor.id,
        created_at=utc_stamp(), status=HandoffStatus.QUEUED,
        settings=validate_conversion_settings(payload.settings),
    )
    session.add(handoff)
    session.flush()
    record(session, 'handoff.queued', handoff.id, actor.id)
    session.commit()
    notify_worker()
    return handoff_info(handoff, None)


def export_table(session, handoff_id, key, actor):
    handoff, review = get_handoff(session, handoff_id, actor)
    if handoff.superseded_by is not None:
        raise HTTPException(409, f'该交接已被复核替换，请下载交接 {handoff.superseded_by}')
    if review is None or review.decision != Decision.APPROVED:
        raise HTTPException(409, '复核通过后可下载正式 CSV')
    if not handoff.result or not handoff.result['handoff']['ready']:
        raise HTTPException(409, '本次交接已暂缓，没有正式下载表')
    for table in handoff.result['tables']:
        if table['key'] == key:
            return handoff, table
    raise LookupError('找不到结果表')


def review_info(review):
    return {
        'handoff_id': review.handoff_id, 'decision': review.decision, 'note': review.note,
        'created_at': review.created_at, 'basis': review.basis,
    }


def submit_review(session, handoff_id, payload, actor):
    handoff = session.scalar(select(Handoff).where(Handoff.id == handoff_id).with_for_update())
    if not handoff:
        raise LookupError('找不到交接')
    if handoff.status != HandoffStatus.COMPLETED or session.get(HandoffReview, handoff_id):
        raise HTTPException(409, '仅可复核一次，且计算必须成功完成')
    if handoff.superseded_by is not None:
        raise HTTPException(409, '该交接已被复核替换，不能再单独复核')
    ready = bool((handoff.result or {}).get('handoff', {}).get('ready'))
    if payload.decision == Decision.APPROVED and not ready:
        raise HTTPException(409, '本次交接已暂缓，不能批准；请补足合格链后重新复核')
    basis = _basis(handoff)
    review = HandoffReview(
        handoff_id=handoff_id, reviewer_id=actor.id, decision=payload.decision,
        note=payload.note, created_at=utc_stamp(), basis=basis,
    )
    session.add(review)
    if payload.decision == Decision.APPROVED and handoff.previous_handoff_id is not None:
        previous = session.get(Handoff, handoff.previous_handoff_id)
        if previous is not None:
            previous.superseded_by = handoff.id
            record(session, 'handoff.superseded', previous.id, actor.id)
    record(session, 'handoff.' + payload.decision, handoff_id, actor.id)
    session.commit()
    return {'handoff_id': handoff_id, 'decision': review.decision, 'note': review.note,
            'created_at': review.created_at}


def _basis(handoff):
    """Freeze the verdict and its basis as issued, independent of later revisions."""

    result = handoff.result or {}
    handoff_block = result.get('handoff', {})
    return json.loads(json.dumps({
        'handoff_id': handoff.id, 'label': handoff.label, 'anchor_instrument': handoff.anchor_instrument,
        'base_run_id': handoff.base_run_id, 'previous_handoff_id': handoff.previous_handoff_id,
        'settings': handoff.settings, 'ready': handoff_block.get('ready'),
        'reasons': handoff_block.get('reasons', []), 'rows': handoff_block.get('rows', []),
        'metrics': result.get('metrics', {}), 'completed_at': utc_stamp(),
    }, ensure_ascii=False, sort_keys=True))


def claim_handoffs(session):
    """Return the next queued handoff with its merged input ledgers."""

    handoff = session.scalar(select(Handoff).where(Handoff.status == HandoffStatus.QUEUED)
                             .order_by(Handoff.id).limit(1).with_for_update(skip_locked=True))
    if handoff is None:
        return None
    handoff.status = HandoffStatus.RUNNING
    run = session.get(Run, handoff.base_run_id)
    groups = [session.get(PairedDataset, dataset_id).rows for dataset_id in handoff.dataset_ids]
    previous = session.get(Handoff, handoff.previous_handoff_id) if handoff.previous_handoff_id else None
    previous_block = None if previous is None else {
        'handoff_id': previous.id, 'rows': previous.result['handoff']['rows'],
        'ready': previous.result['handoff']['ready'],
    }
    record(session, 'handoff.started', handoff.id)
    return {
        'handoff_id': handoff.id, 'groups': groups, 'settings': handoff.settings,
        'base_result': run.result, 'anchor': handoff.anchor_instrument, 'previous': previous_block,
        'screening': float(run.settings.get('screening_mpa', 4.2)),
    }


def execute_handoff(job):
    import logging

    from src.config.database import Session

    handoff_id = job['handoff_id']
    result, error_message = None, None
    try:
        rows = merge_pair_rows(job['groups'])
        result = analyze_conversion(
            rows, job['base_result'], job['anchor'], job['settings'],
            screening=job['screening'], previous_handoff=job['previous'],
        )
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except ValueError as error:
        result, error_message = None, str(error)
    except Exception:
        logging.getLogger(__name__).exception('换算交接 %s 失败', handoff_id)
        result, error_message = None, '换算遇到内部错误，请检查对照台账后重试。'
    with Session.begin() as session:
        handoff = session.get(Handoff, handoff_id)
        handoff.status = HandoffStatus.FAILED if error_message is not None else HandoffStatus.COMPLETED
        handoff.result = result
        handoff.error = error_message
        record(session, 'handoff.' + handoff.status, handoff.id)
