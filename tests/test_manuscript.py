from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]

def test_bilingual_sections_citations_and_shared_equations_match():
    en={p.name:p.read_text() for p in (ROOT/'sections/en').glob('*.tex')}
    zh={p.name:p.read_text() for p in (ROOT/'sections/zh').glob('*.tex')}
    assert set(en)==set(zh)
    bib=(ROOT/'references.bib').read_text()
    available=set(re.findall(r'@\w+\{([^,]+),',bib))
    for name in en:
        citations=lambda s:set(k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',s) for k in group.split(','))
        assert citations(en[name])==citations(zh[name]),name
        assert citations(en[name])<=available,name
        shared=lambda s:re.findall(r'\\input\{(shared/[^}]+)\}',s)
        assert shared(en[name])==shared(zh[name]),name

def test_chinese_control_sequences_have_clear_boundaries():
    for p in [ROOT/'main_zh.tex',*list((ROOT/'sections/zh').glob('*.tex'))]:
        assert not re.search(r'\\[A-Za-z]+[\u3400-\u9fff]',p.read_text()),p.name

def test_original_transformer_citation_year_and_refinement_activation():
    bib=(ROOT/'references.bib').read_text();attention=bib[bib.index('{attention,'):]
    assert re.search(r'year\s*=\s*\{2017\}',attention)
    eq=(ROOT/'shared/eq_refinement.tex').read_text()
    assert 'A_{\\rm win}' in eq and 'A_{\\rm var}' in eq
    assert r'\exists k\in\mathcal C_{j,c}:\eta_k=0' in (ROOT/'shared/eq_attention.tex').read_text()


def test_paper_uses_academic_names_instead_of_internal_experiment_codes():
    papers=[*list((ROOT/'sections').rglob('*.tex')),*list((ROOT/'shared').glob('*.tex')),*list((ROOT/'tables/generated').glob('*.tex')),ROOT/'main_en.tex',ROOT/'main_zh.tex']
    for p in papers:
        assert not re.search(r'R3|E[0-8]|H_GLOBAL',p.read_text()),p
    assert r'\newcommand{\Model}{EviRefine}' in (ROOT/'shared/macros.tex').read_text()
    for lang in ['en','zh']:
        intro=(ROOT/f'sections/{lang}/introduction.tex').read_text()
        method=(ROOT/f'sections/{lang}/method.tex').read_text()
        for name in ['TVCE','MPER','PEGR','OCRR']:
            assert name in intro and name in method,(lang,name)


def test_display_equations_have_one_expression_per_line():
    for p in (ROOT/'shared').glob('eq_*.tex'):
        assert r'\qquad' not in p.read_text(),p
