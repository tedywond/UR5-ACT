import sys
sys.path.append('/opt/language-table')

from language_table.environments.language_table import LanguageTable
from language_table.environments.blocks import LanguageTableBlockVariants

if __name__ == '__main__':
    env = LanguageTable(LanguageTableBlockVariants.BLOCK_8)
    env.reset()
    img = env.render()