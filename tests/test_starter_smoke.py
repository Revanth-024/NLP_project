"""Setup smoke; independent of unfinished algorithm outputs."""
from pathlib import Path
import ast,json,re
ROOT=Path(__file__).resolve().parents[1]
def test_modules_and_todo_inventory():
    ids=[]
    for file in (ROOT/'src').glob('*.py'):
        text=file.read_text(encoding='utf-8-sig');ast.parse(text)
        ids+=re.findall(r'STUDENT TODO (\d+\.\d+)',text)
    assert len(set(ids))==49
    assert not (ROOT/'docs/INSTRUCTOR_GUIDE.md').exists()
def test_supplied_data_and_pages():
    assert len(list((ROOT/'pages').glob('*.py')))==10
    assert len(list((ROOT/'notebooks').glob('*.ipynb')))==8
    assert len(list((ROOT/'data/academic_documents').glob('*.md')))==14
    assert json.loads((ROOT/'data/university_facts.json').read_text())['institution']['name']=='Hindu College of Engineering'
