from fastapi import APIRouter, Query

from src.controllers.dependencies import Actor, Db, Writer
from src.schemas.handoffs import PairedDatasetImport, PairedDatasetInfo
from src.services.paired_catalog import get_dataset, import_dataset, list_datasets, overview

router = APIRouter(prefix='/paired-datasets')


@router.get('', response_model=list[PairedDatasetInfo])
def datasets(session: Db, actor: Actor):
    return list_datasets(session)


@router.get('/{dataset_id}')
def detail(dataset_id: int, session: Db, actor: Actor, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    dataset = get_dataset(session, dataset_id)
    return {**overview(dataset), 'rows': dataset.rows[offset:offset + limit]}


@router.post('', response_model=PairedDatasetInfo, status_code=201)
def insert(payload: PairedDatasetImport, session: Db, actor: Writer):
    return import_dataset(session, payload, actor.id)
