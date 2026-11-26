import json

from sqlalchemy import func, select

from migrations.bootstrap import bootstrap
from src.config.database import Session
from src.constants.workflow import Role
from src.domain.project import SPEC, demo_pairs, demo_rows
from src.models.account import Account
from src.models.dataset import Dataset
from src.models.paired_dataset import PairedDataset
from src.services.catalog import add_dataset
from src.services.paired_catalog import add_dataset as add_paired_dataset
from src.utils.security import hash_password


def seed():
    with Session.begin() as session:
        owner = None
        for role in Role:
            email = f'{role.value}@demo.local'
            account = session.scalar(select(Account).where(Account.email == email))
            if account is None:
                account = Account(email=email, role=role, password_hash=hash_password('Demo-2611!'))
                session.add(account)
                session.flush()
            if role == Role.ADMIN:
                owner = account.id
        name = SPEC['dataset_name']
        dataset = session.scalar(select(Dataset).where(Dataset.name == name))
        if not dataset:
            dataset = add_dataset(session, name, demo_rows(), owner)
        paired_name = SPEC['paired_dataset_name']
        paired = session.scalar(select(PairedDataset).where(PairedDataset.name == paired_name))
        if not paired:
            paired = add_paired_dataset(session, paired_name, demo_pairs(), owner)
        return {'accounts': session.scalar(select(func.count()).select_from(Account)),
                'datasets': session.scalar(select(func.count()).select_from(Dataset)),
                'paired_datasets': session.scalar(select(func.count()).select_from(PairedDataset)),
                'demo_rows': len(dataset.rows), 'checksum': dataset.checksum,
                'paired_rows': len(paired.rows), 'paired_checksum': paired.checksum}


if __name__ == '__main__':
    bootstrap()
    print(json.dumps(seed(), ensure_ascii=False))
