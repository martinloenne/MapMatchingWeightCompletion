"""empty message

Revision ID: 7aa146c7cc37
Revises: 
Create Date: 2020-12-16 14:06:04.021403

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '7aa146c7cc37'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('osmnode',
    sa.Column('node_id', sa.BigInteger(), nullable=False),
    sa.Column('latitude', sa.Float(), nullable=True),
    sa.Column('longitude', sa.Float(), nullable=True),
    sa.PrimaryKeyConstraint('node_id')
    )
    op.create_foreign_key(None, 'osmnodeonway', 'osmnode', ['node_id'], ['node_id'])
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'osmnodeonway', type_='foreignkey')
    op.drop_table('osmnode')
    # ### end Alembic commands ###