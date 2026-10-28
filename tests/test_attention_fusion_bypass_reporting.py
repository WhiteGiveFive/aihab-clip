"""Two-primary reporting, explicit original reuse and fixed-class diagnostics."""
import copy
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from multimodal import attention_fusion_bypass_reporting as reporting
from multimodal.attention_fusion_bypass_comparison import BYPASSES, declare_contrasts
from multimodal.attention_fusion_bypass_protocol import build_matrix, expected_config
from multimodal.attention_fusion_comparison import compare_selected, METRICS
from multimodal.attention_fusion_training import classification_metrics
from tools.prepare_attention_fusion_protocol import build_matrix as original_matrix

ROOT=Path(__file__).resolve().parents[1]


def analysis_fixture():
    old=yaml.safe_load((ROOT/'configs/multimodal_attention_fusion.yaml').read_text())
    config=expected_config(old)
    config['metrics']['bootstrap']['replicates']=13
    matrix=build_matrix(config)
    declaration=declare_contrasts(config,matrix)
    labels=np.array([0,0,0,1,1])
    arms,records=[],[]
    for index,arm_spec in enumerate([*original_matrix(old)['candidate_geo_arms'],*matrix['candidate_geo_arms']]):
        per_seed,histories={},{}
        for seed in range(1,5):
            prediction=np.array([0,0,(index+seed)%2,0,1])
            logits=np.zeros((5,18));logits[np.arange(5),prediction]=2
            metrics,predictions=classification_metrics(labels,logits)
            fit_id=f"{arm_spec['arm_id']}_seed{seed}"
            per_seed[str(seed)]={'metrics':metrics,'fit_id':fit_id,'parameters':1000+index*100,
                'runtime_seconds':10+seed+index,'peak_cuda_bytes':2**20*(seed+1),
                'peak_process_rss_bytes':2**30}
            histories[str(seed)]=[{'epoch':e,'optimization_loss':2/e,
                'train':{**metrics,'loss':2/e},'validation':{**metrics,'loss':2.5/e}} for e in (1,2,3)]
            records.append({'fit_id':fit_id,'predictions':{**predictions,'label':labels,
                'row_uid':np.array([f'r{i}' for i in range(5)]),'plot_idx':np.array(['a','a','b','c','c'])}})
        arms.append({'arm_id':arm_spec['arm_id'],'candidate_id':arm_spec['candidate_id'],
            'geo_mode':arm_spec['geo_preprocessing'],'learning_rate':0.001,'selected_learning_rate':0.001,
            'per_seed':per_seed,'histories_by_seed':histories,
            'study_role':'new_bypass' if arm_spec['candidate_id'] in BYPASSES else 'reused_original',
            'mean':{m:float(np.mean([x['metrics'][m] for x in per_seed.values()])) for m in METRICS},
            'sample_std':{m:float(np.std([x['metrics'][m] for x in per_seed.values()],ddof=1)) for m in METRICS}})
    contrasts,_,_=compare_selected(config,arms,records,declaration)
    all_rates=[{**copy.deepcopy(a),'learning_rate':lr} for a in arms for lr in (0.0001,0.0003,0.001)]
    return {'schema_version':'attention_fusion.bypass_comparison.v1','study_id':'synthetic_bypass_reporting',
        'class_order':list(range(18)),'class_labels':[f'Habitat {i}' for i in range(18)],
        'training_seeds':[1,2,3,4],'validation_rows':5,'validation_plots':3,
        'validation_class_support_images':[3,2]+[0]*16,'validation_class_support_plots':[2,1]+[0]*16,
        'selected_arms':arms,'lr_candidates':[a for a in all_rates if a['candidate_id'] in BYPASSES],
        'reused_lr_candidates':[a for a in all_rates if a['candidate_id'] not in BYPASSES],
        'primary_contrast_ids':declaration['primary_contrast_ids'],'contrasts':contrasts,
        'counts':{'development_fits':72,'reused_development_fits':228,'total_authenticated_fits':300,
            'new_selected_arms':6,'reused_selected_arms':19,'selected_arms':25,'selected_checkpoints':24,
            'reused_selected_checkpoints':76,'contrasts':37},'source_study':config['source_study']}


def test_complete_report_retains_both_primaries_context_all_seed_metrics_and_new_confusions(tmp_path,monkeypatch):
    analysis=analysis_fixture()
    def save(fig,stem,dpi):
        stem.parent.mkdir(parents=True,exist_ok=True)
        for ext in ('png','pdf'):
            stem.with_suffix('.'+ext).write_text('synthetic figure emission\n')
        return []
    def draw(cm,labels,sink):
        assert cm.shape==(18,18) and len(labels)==18
        denominator=cm.sum(axis=1,keepdims=True).copy();denominator[denominator==0]=1
        for normalized,values in ((False,cm),(True,cm/denominator)):
            fig,_=plt.subplots()
            try:
                sink(fig,values,normalized)
            finally:
                plt.close(fig)
    monkeypatch.setattr(reporting,'_save',save)
    monkeypatch.setattr(reporting.original,'_save',save)
    monkeypatch.setattr(reporting,'draw_cm',draw)
    destination=tmp_path/'report'
    result=reporting.generate_report(analysis,destination,png_dpi=72)
    assert result['primary_contrast_ids']==analysis['primary_contrast_ids']
    assert result['counts']['selected_checkpoints']==24
    text=(destination/'report.md').read_text()
    assert 'Two primary comparisons' in text and 'after observing the original M3' in text
    assert 'not simultaneous intervals' in text and 'do not identify an isolated' in text
    assert '72 new fits' in text and '228 original fits' in text and '25 selected arms' in text
    assert text.count('declared category **')==2
    assert 'not an ensemble' in text and 'recall is NA' in text
    for stem in ('all_new_fits','reused_fits','learning_rates','reused_learning_rates','selected_arms','paired_contrasts'):
        assert f'../tables/{stem}.csv' in text
    for a in analysis['selected_arms']:
        assert (destination/f"figures/learning_curves/{a['arm_id']}.png").exists()
        folder=destination/'figures/confusion'/a['arm_id']
        assert folder.exists()==(a['candidate_id'] in BYPASSES)
        if folder.exists():
            assert len(list(folder.glob('*.png')))==10
    per_seed=(destination/'per_seed_metrics.md').read_text()
    assert all(value['fit_id'] in per_seed for arm in analysis['selected_arms'] for value in arm['per_seed'].values())
    with (destination/'tables/contrast_per_class.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==37*18
    assert all(r['mean_recall_difference']=='' for r in rows if r['validation_images']=='0')
    assert len([r for r in rows if r['contrast_id'] in analysis['primary_contrast_ids']])==36
    assert not plt.get_fignums()
    before={str(p):p.stat().st_mtime_ns for p in destination.rglob('*') if p.is_file()}
    assert reporting.generate_report(analysis,destination,png_dpi=72)==result
    assert before=={str(p):p.stat().st_mtime_ns for p in destination.rglob('*') if p.is_file()}
    changed=copy.deepcopy(analysis);changed['study_id']='different'
    with pytest.raises(ValueError,match='changed'):
        reporting.generate_report(changed,destination,png_dpi=72)


@pytest.mark.parametrize('violation',['missing_primary','missing_original_arm','missing_seed','confusion_support'])
def test_report_rejects_missing_evidence_before_creating_output(tmp_path,violation):
    analysis=analysis_fixture()
    if violation=='missing_primary':
        analysis['primary_contrast_ids'].pop()
    elif violation=='missing_original_arm':
        analysis['selected_arms']=[a for a in analysis['selected_arms'] if a['candidate_id']!='geo_only']
    elif violation=='missing_seed':
        analysis['selected_arms'][0]['per_seed'].pop('4')
    else:
        analysis['selected_arms'][0]['per_seed']['1']['metrics']['confusion_matrix'][0][0]+=1
    with pytest.raises(ValueError):
        reporting.generate_report(analysis,tmp_path/'report')
    assert not list(tmp_path.iterdir())


def test_cli_report_authenticates_bypass_identity_and_uses_separate_destination(tmp_path,monkeypatch,capsys):
    from tools import analyze_attention_fusion_bypass_study as cli
    config={'study_id':'bypass','paths':{'artifact_root':'extension'}}
    matrix={'example':True}
    monkeypatch.setattr(cli,'REPO_ROOT',tmp_path)
    monkeypatch.setattr(cli,'read_frozen_plan',lambda *a,**k:(config,matrix,{}))
    monkeypatch.setattr(cli,'authenticated_analysis',lambda *a:{'study_id':'wrong','provenance':{
        'config_sha256':cli.content_hash(config),'matrix_sha256':cli.content_hash(matrix)}})
    with pytest.raises(ValueError,match='different frozen bypass'):
        cli.main(['report'])
    with pytest.raises(ValueError,match='separate study directory'):
        cli.main(['report','--analysis-dir',str(tmp_path/'original')])
