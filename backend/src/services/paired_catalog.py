import hashlib
import json

from sqlalchemy import select

from src.domain.project import SPEC, normalize_pairs
from src.models.paired_dataset import PairedDataset
from src.services.audit import record
from src.utils.csv_files import read_rows


def overview(dataset):
    instruments = sorted({row['instrument'] for row in dataset.rows})
    return {
        'id': dataset.id, 'name': dataset.name, 'checksum': dataset.checksum,
        'count': len(dataset.rows), 'strips': len({row['id'] for row in dataset.rows}),
        'instruments': instruments,
    }


def list_datasets(session):
    return [overview(dataset) for dataset in session.scalars(select(PairedDataset).order_by(PairedDataset.id.desc()))]


def get_dataset(session, dataset_id):
    dataset = session.get(PairedDataset, dataset_id)
    if dataset is None:
        raise LookupError('找不到对照台账')
    return dataset


def add_dataset(session, name, rows, owner):
    normalized = normalize_pairs(rows)
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    dataset = PairedDataset(name=name, rows=normalized,
                            checksum=hashlib.sha256(encoded).hexdigest(), created_by=owner)
    session.add(dataset)
    session.flush()
    return dataset


def import_dataset(session, payload, owner):
    dataset = add_dataset(session, payload.name, read_rows(payload.csv, SPEC['paired_columns']), owner)
    record(session, 'paired_dataset.imported', dataset.id, owner)
    session.commit()
    return overview(dataset)
