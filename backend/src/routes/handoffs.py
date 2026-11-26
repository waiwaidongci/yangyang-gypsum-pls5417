from fastapi import APIRouter
from fastapi.responses import Response

from src.controllers.dependencies import Actor, Admin, Db, Writer
from src.schemas.handoffs import HandoffCreate, HandoffInfo, HandoffReviewCreate
from src.services import handoffs
from src.utils.csv_files import write_rows

router = APIRouter(prefix='/handoffs')


@router.get('', response_model=list[HandoffInfo])
def list_all(session: Db, actor: Actor):
    return handoffs.list_handoffs(session, actor)


@router.post('', response_model=HandoffInfo, status_code=202)
def create(payload: HandoffCreate, session: Db, actor: Writer):
    return handoffs.create_handoff(session, payload, actor)


@router.get('/{handoff_id}')
def detail(handoff_id: int, session: Db, actor: Actor):
    handoff, review = handoffs.get_handoff(session, handoff_id, actor)
    return {**handoffs.handoff_info(handoff, review), 'result': handoff.result, 'error': handoff.error,
            'review': handoffs.review_info(review) if review else None}


@router.post('/{handoff_id}/review', status_code=201)
def review(handoff_id: int, payload: HandoffReviewCreate, session: Db, actor: Admin):
    return handoffs.submit_review(session, handoff_id, payload, actor)


@router.get('/{handoff_id}/export/{key}')
def export(handoff_id: int, key: str, session: Db, actor: Actor):
    handoff, table = handoffs.export_table(session, handoff_id, key, actor)
    return Response(write_rows(table['columns'], table['rows']), media_type='text/csv',
                    headers={'Content-Disposition':
                             f'attachment; filename="handoff-{handoff_id}-{table["key"]}.csv"'})
