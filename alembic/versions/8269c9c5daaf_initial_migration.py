"""Initial migration

Revision ID: 8269c9c5daaf
Revises: 
Create Date: 2026-07-24 00:48:15.110394

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8269c9c5daaf'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'simulation_results',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('tenant_id', sa.String(), index=True),
        sa.Column('run_type', sa.String()),
        sa.Column('parameters', sa.JSON()),
        sa.Column('results', sa.JSON()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'))
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_simulation_results_tenant_id'), table_name='simulation_results')
    op.drop_index(op.f('ix_simulation_results_id'), table_name='simulation_results')
    op.drop_table('simulation_results')
