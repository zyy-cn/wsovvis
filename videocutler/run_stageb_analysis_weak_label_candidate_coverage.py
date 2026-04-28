#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
from collections import Counter,defaultdict

def js(p):
    with open(p,'r',encoding='utf-8') as f:return json.load(f)
def itj(p):
    with open(p,'r',encoding='utf-8') as f:
        for l in f:
            l=l.strip()
            if not l: continue
            try:r=json.loads(l)
            except Exception:continue
            if isinstance(r,dict):yield r
def si(x):
    try:return int(x)
    except Exception:return None
def vid(r):
    for k in ['video_id','clip_id','id','video','video_idx','dataset_video_id']:
        if k in r and r[k] is not None:return str(r[k])
    t=r.get('trajectory_id') or r.get('track_id')
    if isinstance(t,str) and ':' in t:
        for p in reversed(t.split(':')[:-1]):
            if p.isdigit():return p
    return None
def iset(x):
    o=set()
    if x is None:return o
    if isinstance(x,dict):
        for k,v in x.items():
            for z in [k,v]:
                iz=si(z)
                if iz is not None:o.add(iz)
        return o
    if isinstance(x,(list,tuple,set)):
        for v in x:
            iv=si(v)
            if iv is not None:o.add(iv)
        return o
    iv=si(x)
    if iv is not None:o.add(iv)
    return o
def classes(r):
    keys=['observed_raw_ids','yprime_raw_ids','weak_raw_ids','positive_raw_ids','candidate_ids_known','candidate_raw_ids','category_ids','classes','labels','raw_ids','gt_raw_ids','class_ids','positive_ids','base_observed_raw_ids','known_raw_ids']
    s=set()
    for k in keys:
        if k in r:s|=iset(r.get(k))
    return s
def first(xs):
    for x in xs:
        if x and Path(x).exists():return Path(x)
    return None
def load_weak(p):
    d=js(p); out={}
    def add(r,forced=None):
        if not isinstance(r,dict):return
        v=forced or vid(r)
        if v is None:return
        out.setdefault(v,set()).update(classes(r))
    if isinstance(d,list):
        for r in d:add(r)
    elif isinstance(d,dict):
        used=False
        for k in ['records','weak_labels','items','videos','clips','data']:
            if isinstance(d.get(k),list):
                used=True
                for r in d[k]:add(r)
        if not used or not out:
            for k,v in d.items():
                if k in ['meta','metadata','summary','schema','version']:continue
                if isinstance(v,dict):add(v,str(k))
                elif isinstance(v,list):
                    if all(not isinstance(x,dict) for x in v):out[str(k)]=iset(v)
                    else:
                        for r in v:add(r,str(k))
    return out
def load_split(p):
    if not p or not p.exists():return set(),set(),{'status':'missing'}
    d=js(p); b=set(); n=set()
    def abs1(obj,ks):
        s=set()
        if isinstance(obj,dict):
            for k in ks:
                if k in obj:s|=iset(obj[k])
        return s
    b|=abs1(d,['base_raw_ids','base_ids','base','base_categories','official_base_raw_ids'])
    n|=abs1(d,['novel_raw_ids','novel_ids','novel','novel_categories','official_novel_raw_ids'])
    if isinstance(d,dict):
        for ck in ['split','splits','official_split','lvvis_official_split']:
            if isinstance(d.get(ck),dict):
                b|=abs1(d[ck],['base_raw_ids','base_ids','base']); n|=abs1(d[ck],['novel_raw_ids','novel_ids','novel'])
    return b,n,{'status':'loaded','path':str(p),'base_count':len(b),'novel_count':len(n)}
def load_gt(p,base,base_only):
    if not p or not p.exists():return {}
    d=js(p); anns=d.get('annotations',[]) if isinstance(d,dict) else d
    out=defaultdict(set)
    for a in anns:
        if not isinstance(a,dict):continue
        v=vid(a); c=None
        for k in ['category_id','raw_id','category_raw_id','class_id']:
            if k in a:c=si(a[k]);break
        if v is None or c is None:continue
        if base_only and base and c not in base:continue
        out[v].add(c)
    return dict(out)
def load_text(p):
    if not p or not p.exists():return set()
    out=set()
    if p.suffix=='.jsonl':
        for r in itj(p):
            out|=classes(r)
            for k in ['raw_id','category_id','class_id','id']:
                if k in r and si(r[k]) is not None:out.add(si(r[k]))
        return out
    d=js(p)
    if isinstance(d,dict):
        out|=iset(d.keys()); out|=classes(d)
        for k in ['prototypes','text_prototypes','classes','categories','records','items']:
            v=d.get(k)
            if isinstance(v,dict):out|=iset(v.keys())
            elif isinstance(v,list):
                for r in v:
                    if isinstance(r,dict):
                        out|=classes(r)
                        for kk in ['raw_id','category_id','class_id','id']:
                            if kk in r and si(r[kk]) is not None:out.add(si(r[kk]))
    return out
def scan_traj(p):
    c=defaultdict(int); total=0; keys=set(); bad=0
    for r in itj(p):
        total+=1
        if total<=10:keys.update(r.keys())
        v=vid(r)
        if v is None:bad+=1
        else:c[v]+=1
    return dict(c),total,{'path':str(p),'clip_count':len(c),'total_trajectory_count':total,'sample_keys':sorted(keys),'malformed_no_video_id':bad}
def wjson(p,o):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(o,ensure_ascii=False,indent=2,sort_keys=True),encoding='utf-8')
def wcsv(p,rows,fields):
    p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows:w.writerow({k:r.get(k,'') for k in fields})
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run_root',required=True); ap.add_argument('--runtime_output_root',default='/mnt/sda/zyy/code/wsovvis'); ap.add_argument('--dataset_name',default='lvvis_train_base')
    ap.add_argument('--trajectory_jsonl'); ap.add_argument('--weak_labels_json'); ap.add_argument('--annotation_json'); ap.add_argument('--official_split_json'); ap.add_argument('--text_bank_json')
    ap.add_argument('--base_only',default='true'); ap.add_argument('--top_examples',type=int,default=128)
    a=ap.parse_args(); repo=Path(a.runtime_output_root); run=Path(a.run_root); ds=a.dataset_name; out=run/'analysis'/'weak_label_candidate_coverage'/ds; out.mkdir(parents=True,exist_ok=True)
    paths={
    'trajectory_jsonl':first([a.trajectory_jsonl,f'/home/zyy/code/wsovvis_asserts/exports/{ds}/trajectory_records.jsonl',repo/'exports'/ds/'trajectory_records.jsonl']),
    'weak_labels_json':first([a.weak_labels_json,repo/'codex/outputs/g3_weak_labels/weak_labels/weak_labels_train.json',repo/'codex/outputs/g3_weak_labels/weak_labels_train.json','/mnt/sda/zyy/code/wsovvis/codex/outputs/g3_weak_labels/weak_labels/weak_labels_train.json']),
    'annotation_json':first([a.annotation_json,'/home/zyy/code/wsovvis_asserts/dataset/LV-VIS/annotations/train_instances.json',repo/'videocutler/datasets/LV-VIS/annotations/train_instances.json']),
    'split_json':first([a.official_split_json,repo/'package/reference/lvvis_official_base_novel_split.json']),
    'text_bank_json':first([a.text_bank_json,f'/home/zyy/code/wsovvis_asserts/text_bank/{ds}/text_prototypes.json',f'/home/zyy/code/wsovvis_asserts/text_bank/{ds}/text_prototypes.jsonl',repo/'text_bank'/ds/'text_prototypes.json',repo/'text_bank'/ds/'text_prototypes.jsonl'])}
    miss=[k for k in ['trajectory_jsonl','weak_labels_json'] if not paths[k]]
    if miss:
        s={'status':'FAIL_MISSING_REQUIRED_INPUTS','missing':miss,'resolved_paths':{k:str(v) if v else None for k,v in paths.items()}}; wjson(out/'summary.json',s); print(json.dumps(s,indent=2)); return 2
    base,novel,split=load_split(paths['split_json']); base_only=str(a.base_only).lower() in ['1','true','yes','y']
    gt=load_gt(paths['annotation_json'],base,base_only) if paths['annotation_json'] else {}; text=load_text(paths['text_bank_json']) if paths['text_bank_json'] else set()
    tc,total,tmeta=scan_traj(paths['trajectory_jsonl']); weak=load_weak(paths['weak_labels_json'])
    vids=sorted(set(tc)|set(weak)|set(gt),key=lambda x:(len(x),x)); reason=Counter(); rtraj=Counter(); rows=[]; ex=[]; clsw=Counter(); clsgmiss=Counter(); clsfp=Counter(); vclips=vtraj=0; nw=nb=nt=0
    for v in vids:
        tn=tc.get(v,0); w=set(weak.get(v,set())); g=set(gt.get(v,set()))
        if w:nw+=1
        wb=(w&base) if base_only and base else set(w)
        if wb:nb+=1
        cand=set(wb); mt=set()
        if text:mt={c for c in cand if c not in text}; cand&=text
        if cand:nt+=1
        rs=[]
        if tn<=0:rs.append('no_trajectory_records')
        if not w:rs.append('missing_weak_label_record_or_empty_yprime')
        elif not wb:rs.append('empty_yprime_after_base_filter')
        elif not cand:rs.append('empty_candidate_after_text_prototype_filter')
        valid=tn>0 and bool(cand)
        if valid:
            vclips+=1; vtraj+=tn
            if not rs:rs=['valid']
        if not rs:rs=['unknown_invalid']
        for c in w:clsw[c]+=1
        for c in g-w:clsgmiss[c]+=1
        for c in w-g:clsfp[c]+=1
        for r in rs:reason[r]+=1; rtraj[r]+=tn
        row={'clip_id':v,'trajectory_count':tn,'weak_raw_count':len(w),'weak_after_base_count':len(wb),'candidate_count':len(cand),'gt_base_class_count':len(g),'weak_intersect_gt_base_count':len(w&g),'weak_not_in_gt_base_count':len(w-g),'gt_base_missing_from_weak_count':len(g-w),'missing_text_id_count':len(mt),'reason':';'.join(rs),'is_valid_trainable_clip':valid}
        rows.append(row)
        if not valid and tn>0 and len(ex)<a.top_examples:
            e=dict(row); e.update({'weak_raw_ids':sorted(w)[:50],'weak_after_base_ids':sorted(wb)[:50],'candidate_ids':sorted(cand)[:50],'gt_base_ids':sorted(g)[:50],'missing_text_ids':sorted(mt)[:50]}); ex.append(e)
    ic=sum(1 for r in rows if r['trajectory_count']>0 and not r['is_valid_trainable_clip']); it=sum(r['trajectory_count'] for r in rows if r['trajectory_count']>0 and not r['is_valid_trainable_clip']); ig=[r for r in rows if r['trajectory_count']>0 and not r['is_valid_trainable_clip'] and r['gt_base_class_count']>0]
    rr=[{'reason':r,'clip_count':c,'trajectory_count':rtraj[r],'rate_vs_all_trajectory_clips':c/len(tc) if tc else 0,'rate_vs_all_trajectories':rtraj[r]/total if total else 0} for r,c in reason.most_common()]
    cr=[]
    for c in sorted(set(clsw)|set(clsgmiss)|set(clsfp)):
        cr.append({'raw_id':c,'weak_pair_count':clsw[c],'gt_missing_from_weak_count':clsgmiss[c],'weak_not_in_gt_count':clsfp[c],'weak_false_positive_rate_proxy':(clsfp[c]/clsw[c] if clsw[c] else '')})
    cr.sort(key=lambda r:(r['gt_missing_from_weak_count'],r['weak_pair_count']),reverse=True)
    igr=(len(ig)/ic) if ic else None; itgr=(sum(r['trajectory_count'] for r in ig)/it) if it else None
    verdict='potential_weak_label_generation_or_binding_gap' if ic and igr and igr>0.5 else ('invalid_clips_likely_protocol_filter_or_empty_yprime' if ic else 'valid_phase1_filtering_explained')
    s={'status':'PASS','verdict':verdict,'dataset_name':ds,'resolved_paths':{k:str(v) if v else None for k,v in paths.items()},'split_meta':split,'base_only':base_only,'base_count':len(base),'novel_count':len(novel),'text_bank_available':bool(text),'text_id_count':len(text),'trajectory_record_count':total,'trajectory_clip_count':len(tc),'weak_label_clip_count_raw':len(weak),'weak_label_clip_count_nonempty':nw,'weak_after_base_clip_count_nonempty':nb,'weak_after_text_clip_count_nonempty':nt,'gt_base_clip_count':len(gt),'valid_trainable_clip_count':vclips,'valid_trainable_trajectory_count':vtraj,'invalid_trajectory_clip_count':ic,'invalid_trajectory_count':it,'valid_rate_vs_raw_trajectories':vtraj/total if total else 0,'invalid_rate_vs_raw_trajectories':it/total if total else 0,'invalid_clip_gt_has_base_class_rate':igr,'invalid_traj_gt_has_base_class_rate':itgr,'reason_counts':dict(reason),'reason_trajectory_counts':dict(rtraj),'trajectory_meta':tmeta,'interpretation':{'is_gt_auditable_filter':False,'verdict':verdict,'meaning':'Explains weak-label/candidate validity filtering. It is not GT/auditable trajectory filtering.'}}
    wjson(out/'summary.json',s); wcsv(out/'invalid_reason_summary.csv',rr,['reason','clip_count','trajectory_count','rate_vs_all_trajectory_clips','rate_vs_all_trajectories']); wcsv(out/'clip_weak_candidate_rows.csv',rows,['clip_id','trajectory_count','weak_raw_count','weak_after_base_count','candidate_count','gt_base_class_count','weak_intersect_gt_base_count','weak_not_in_gt_base_count','gt_base_missing_from_weak_count','missing_text_id_count','reason','is_valid_trainable_clip']); wcsv(out/'class_weak_candidate_summary.csv',cr,['raw_id','weak_pair_count','gt_missing_from_weak_count','weak_not_in_gt_count','weak_false_positive_rate_proxy'])
    with open(out/'invalid_clip_examples.jsonl','w',encoding='utf-8') as f:
        for e in ex:f.write(json.dumps(e,ensure_ascii=False)+'\n')
    md=f"""# Weak-label/Candidate Coverage Audit\n\n- status: PASS\n- verdict: {verdict}\n- dataset: {ds}\n- trajectory_record_count: {total}\n- trajectory_clip_count: {len(tc)}\n- weak_label_clip_count_raw: {len(weak)}\n- valid_trainable_clip_count: {vclips}\n- valid_trainable_trajectory_count: {vtraj}\n- invalid_trajectory_clip_count: {ic}\n- invalid_trajectory_count: {it}\n- valid_rate_vs_raw_trajectories: {s['valid_rate_vs_raw_trajectories']}\n- invalid_rate_vs_raw_trajectories: {s['invalid_rate_vs_raw_trajectories']}\n- invalid_clip_gt_has_base_class_rate: {igr}\n- invalid_traj_gt_has_base_class_rate: {itgr}\n- is_gt_auditable_filter: false\n\n## Top reasons\n"""
    for x in rr[:20]:md+=f"- {x['reason']}: clips={x['clip_count']}, trajectories={x['trajectory_count']}\n"
    md+=f"\n## Outputs\n- summary: `{out/'summary.json'}`\n- reasons: `{out/'invalid_reason_summary.csv'}`\n- rows: `{out/'clip_weak_candidate_rows.csv'}`\n- classes: `{out/'class_weak_candidate_summary.csv'}`\n- examples: `{out/'invalid_clip_examples.jsonl'}`\n"
    (out/'WEAK_LABEL_CANDIDATE_COVERAGE_TAKEOVER.md').write_text(md,encoding='utf-8'); print(md); return 0
if __name__=='__main__':raise SystemExit(main())
