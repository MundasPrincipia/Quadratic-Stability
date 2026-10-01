"""Read-only raw S1/S4 IQ -> v16 features -> fold-local ridge heads.

The historical numerical interfaces are used only as regression targets and
as the explicit, fixed measurement-bank design. No cached C or H is fitted.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import scipy
import sklearn
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

from common import HERE, ROOT, RF, sha, now, write_json, write_csv, check_lock, decode
from czrf.passive_quotient.frontends import segmented_fourier_frontend, complex_unit_rows
from czrf.passive_quotient.mechanism import physically_order_segment_major_rows
from czrf.passive_quotient import fit_group_standardizer, projective_intensity_features
from czrf.passive_quotient.sketch import ProjectiveMeasurementBank
from czrf.passive_quotient.harmonic import segmented_harmonic_coefficient_matrix
from czrf.track_a_v2_features import ideal_upchirp
from czrf.oregon_lora_adapter import load_oregon_complex64, detect_oregon_preamble_frames
from czrf.ordered_task_interface import normalized_adapted_matrices, pairwise_task_operators_adapted
from czrf.projective_task_certificate import raw_linear_head, pair_weight_matrix, projective_feature_upper_bound


def inherited_interval_module():
    path = ROOT/'paper/development_track_s_task_singularities_candidate_v10_20260902/src/run_rf_selective_mechanism.py'
    spec = importlib.util.spec_from_file_location('v16_inherited_interval', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def s1(out):
    manifest = ROOT/'configs/track_c_v1/B1_ROW_MANIFEST_V1_20260827_154500.csv'
    with manifest.open(encoding='utf-8', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['source_id']=='S1']
    assert len(rows)==5200
    c = np.empty((len(rows),256),np.complex128)
    labels = np.array([int(r['device']) for r in rows])
    groups = np.array([r['group_id'] for r in rows])
    envs = sorted({r['environment'] for r in rows})
    environment = np.array([envs.index(r['environment']) for r in rows])
    inputs, ledger = [], []
    for path in dict.fromkeys(r['source_path'] for r in rows):
        ids = [i for i,r in enumerate(rows) if r['source_path']==path]
        full = ROOT/path
        digest = sha(full)
        inputs.append(dict(path=path,sha256=digest,bytes=full.stat().st_size,rows=len(ids)))
        with h5py.File(full,'r') as h:
            for start in range(0,len(ids),64):
                pos = np.array(ids[start:start+64])
                rawrows = np.array([int(rows[i]['source_row']) for i in pos])
                order = np.argsort(rawrows)
                stored = np.asarray(h['data'][rawrows[order]])[np.argsort(order)]
                iq = stored[:,:8192]+1j*stored[:,8192:]
                rawlabels = np.asarray(h['label'][0,rawrows[order]]).reshape(-1)[np.argsort(order)]
                if not np.array_equal(np.rint(rawlabels).astype(int),labels[pos]):
                    raise AssertionError('S1 raw labels differ')
                front = segmented_fourier_frontend(iq,segment_count=8,coefficients_per_segment=32)
                # This precision boundary is part of the historical frontend.
                front = np.asarray(front,dtype=np.complex64)
                c[pos] = physically_order_segment_major_rows(front,segment_count=8,coefficients_per_segment=32)
                for j,i in enumerate(pos):
                    ledger.append(dict(source='S1',row_id=int(i),path=path,raw_row=int(rawrows[j]),start_sample='',
                                       raw_record_sha256=hashlib.sha256(np.ascontiguousarray(stored[j]).tobytes()).hexdigest(),
                                       device=int(labels[i]),environment=rows[i]['environment']))
        print('S1 raw rebuild:',path,len(ids),flush=True)
    write_csv(out/'RAW_ROW_LINEAGE.csv',sorted(ledger,key=lambda r:r['row_id']))
    write_json(out/'RAW_SOURCE_HASHES.json',inputs)
    return c,labels,groups,environment


def s4(out):
    meta = ROOT/'runs/s4_oregon/S4_OREGON_FEATURE_CACHE_20260827_032417.h5'
    registry = ROOT/'runs/track_c_v0/M2_SMOKE_HARMONIC_FEATURES_20260827_144025.npz'
    with np.load(registry,allow_pickle=False) as h:
        selected = h['S4_selected_indices']
    with h5py.File(meta,'r') as h:
        role = h['role'][:]; files = h['file_id'][:]; hashes = h['packet_sha256'][:]
        eligible = np.flatnonzero(np.isin(role,[0,1]))
        independent = sorted(i for fid in np.unique(files[eligible]) for i in sorted(eligible[files[eligible]==fid],key=lambda j:hashes[j])[:12])
        assert np.array_equal(selected,independent)
        labels = np.asarray(h['device'][selected],int)
        days = np.asarray(h['day'][selected],int)
        fileids = np.asarray(h['file_id'][selected],int)
        starts = np.asarray(h['start_sample'][selected],int)
        phashes = [decode(z) for z in h['packet_sha256'][selected]]
        paths = [decode(z) for z in h['file_paths'][:]]
    assert len(selected)==480 and len(np.unique(fileids))==40
    c = np.empty((480,128),np.complex128)
    ref = ideal_upchirp(sample_rate_hz=1000000,bandwidth_hz=125000,symbol_samples=1024)
    inventory_path = ROOT/'results/s4_oregon_qualification/S4_OREGON_SOURCE_INVENTORY_20260827_030353.csv'
    qualification = json.loads((ROOT/'results/s4_oregon_qualification/S4_OREGON_SOURCE_QUALIFICATION_20260827_030353.json').read_text())
    source_digest = qualification['inventory']['source_content_sha256']
    with inventory_path.open(encoding='utf-8-sig',newline='') as f:
        inv = list(csv.DictReader(f))
    inputs,ledger,detector_rows = [],[],[]
    for fid in np.unique(fileids):
        ids = np.flatnonzero(fileids==fid)
        raw = ROOT/'ieee_dataport/Datasets/Oregon_LoRa'/paths[fid]
        digest = sha(raw)
        # The source inventory is checked by matching its literal path suffix.
        matches = [r for r in inv if any(str(v).replace('\\','/').endswith(paths[fid].replace('\\','/')) for v in r.values())]
        if len(matches)!=1 or digest not in matches[0].values():
            raise AssertionError('S4 original source digest mismatch: '+str(raw))
        iq = load_oregon_complex64(raw)
        detected = detect_oregon_preamble_frames(iq)
        if isinstance(detected,dict):
            raise TypeError('inspect detector return contract')
        found = {int(r.start_sample) for r in detected}
        for i in ids:
            start = int(starts[i])
            if start not in found:
                raise AssertionError('selected S4 detector start not reproduced')
            frame = np.asarray(iq[start:start+8192],np.complex128)
            packet = hashlib.sha256(np.asarray(iq[start:start+8192],dtype=np.complex64).tobytes()).hexdigest()
            identity_hash = hashlib.sha256(f'S4_OREGON_LIMITED_V1|{source_digest}|{digest}|{start}'.encode()).hexdigest()
            if identity_hash!=phashes[i]:
                raise AssertionError('S4 packet digest mismatch')
            coefficients = segmented_harmonic_coefficient_matrix(frame.reshape(8,1024),frequency_bins=16,
                                reference=ref,remove_dominant_tone=True)
            unit = coefficients.reshape(-1)/np.linalg.norm(coefficients.reshape(-1))
            # Preserve the historical real-coordinate conversion and renormalization.
            real = np.concatenate((unit.real,unit.imag)).astype(np.float32).astype(np.float64)
            c[i] = complex_unit_rows((real[:128]+1j*real[128:])[None,:])[0]
            ledger.append(dict(source='S4',row_id=int(i),path=raw.relative_to(ROOT).as_posix(),
                               raw_row=int(selected[i]),start_sample=start,raw_record_sha256=packet,
                               device=int(labels[i]),environment=int(days[i])))
        inputs.append(dict(path=raw.relative_to(ROOT).as_posix(),sha256=digest,bytes=raw.stat().st_size,rows=len(ids)))
        detector_rows.append(dict(file_id=int(fid),detected_frames=len(detected),selected_frames=len(ids),all_starts_match=True))
        print('S4 raw detector/frontend:',int(fid),len(ids),flush=True)
    write_csv(out/'RAW_ROW_LINEAGE.csv',sorted(ledger,key=lambda r:r['row_id']))
    write_json(out/'RAW_SOURCE_HASHES.json',inputs)
    write_csv(out/'DETECTOR_REPLAY.csv',detector_rows)
    return c,labels,np.array([f'file:{i}' for i in fileids]),days


def run(output):
    check_lock()
    output.mkdir(parents=True,exist_ok=False)
    mod = inherited_interval_module()
    flags = np.load(RF/'FLAGS.npz',allow_pickle=False)
    np.savez_compressed(output/'FLAGS.npz',**{k:flags[k] for k in flags.files})
    baseline,totals = [],dict(cells=0,accepted=0,label_correct_accepted=0,accept_mismatches=0)
    for source,loader in [('S1',s1),('S4',s4)]:
        dest = output/source; dest.mkdir()
        c,labels,groups,environment = loader(dest)
        old = np.load(RF/source/'SOURCE_INPUTS.npz',allow_pickle=False)
        for key,value in [('labels',labels),('groups',groups),('environment',environment)]:
            if not np.array_equal(value,old[key]):
                raise AssertionError(source+' '+key+' differs')
        cerror = float(np.max(np.abs(c-old['C'])))
        if cerror>1e-10:
            raise AssertionError(f'{source} C reconstruction mismatch {cerror}')
        a = old['measurement_vectors']
        bank = ProjectiveMeasurementBank(a,20260827,a.shape[1],a.shape[0],sha(RF/source/'SOURCE_INPUTS.npz'))
        features = projective_intensity_features(c,bank)
        bound = projective_feature_upper_bound(a.shape[1],a.shape[0])
        np.savez_compressed(dest/'SOURCE_INPUTS.npz',C=c,labels=labels,groups=groups,environment=environment,
                            row_ids=np.arange(len(c)),measurement_vectors=a,features=features)
        for hp in sorted((RF/source).glob('*/COMMON_HEAD.npz')):
            refhead = np.load(hp,allow_pickle=False)
            train,ids = refhead['train_ids'],refhead['validation_ids']
            assert len(np.intersect1d(train,ids))==0
            assert len(np.unique(environment[ids]))==1
            assert set(train)==set(np.flatnonzero(environment!=environment[ids[0]]))
            state = fit_group_standardizer(features[train],groups[train],forbidden_groups=groups[ids])
            model = RidgeClassifier(alpha=1.,class_weight=None).fit(state.transform(features[train]),labels[train])
            w,b = raw_linear_head(model.coef_,model.intercept_,state.mean,state.scale)
            _,pairs = pair_weight_matrix(w)
            scores = features[ids]@w.T+b
            h = pairwise_task_operators_adapted(a,w,bound,c.shape[1]//8,8,basis=np.eye(8))
            fold = dest/hp.parent.name; fold.mkdir()
            np.savez_compressed(fold/'COMMON_HEAD.npz',train_ids=train,validation_ids=ids,classes=model.classes_,
                raw_weights=w,raw_intercept=b,standardizer_mean=state.mean,standardizer_scale=state.scale,
                class_operators_physical=h,pairs=pairs,full_scores=scores)
            for flag in flags.files:
                oldrow = np.load(hp.parent/f'{flag}_ROW_ARRAYS.npz',allow_pickle=False)
                adapted = normalized_adapted_matrices(c[ids],c.shape[1]//8,8,basis=flags[flag])
                adapted_bank = normalized_adapted_matrices(a,c.shape[1]//8,8,basis=flags[flag])
                calc = mod.numeric_interval(adapted,adapted_bank,w,b,pairs,bound)
                mins,_,_ = mod.pairwise_minimum(calc['center'],calc['radius']+calc['envelope'],pairs,10)
                candidate = np.argmax(mins,axis=1)
                accepted = mins[np.arange(len(ids)),candidate]>1e-12
                mismatches = int(np.sum(accepted!=oldrow['exact_accepted']))
                scoreerror = float(np.max(np.abs(scores-oldrow['full_scores'])))
                if scoreerror>1e-8 or mismatches:
                    raise AssertionError(f'baseline mismatch {source}/{hp.parent.name}/{flag}: {scoreerror}, {mismatches}')
                np.savez_compressed(fold/f'{flag}_BASELINE.npz',row_ids=ids,center_class=calc['center_class'],
                    pair_center=calc['center'],pair_radius=calc['radius'],envelope=calc['envelope'],
                    accepted=accepted,candidate=candidate,full_scores=scores)
                correct = int(np.sum(accepted&(model.classes_[candidate]==labels[ids])))
                totals['cells']+=len(ids); totals['accepted']+=int(accepted.sum())
                totals['label_correct_accepted']+=correct; totals['accept_mismatches']+=mismatches
                baseline.append(dict(source=source,fold=hp.parent.name,flag=flag,rows=len(ids),C_max_error=cerror,
                    score_max_error=scoreerror,operator_max_error=float(np.max(np.abs(h-refhead['class_operators_physical']))),
                    accept_mismatches=mismatches,accepted=int(accepted.sum()),label_correct_accepted=correct))
            print('Refitted baseline:',source,hp.parent.name,flush=True)
    assert totals['cells']==34080 and totals['accepted']==6049 and totals['label_correct_accepted']==4913
    write_csv(output/'BASELINE_REPLAY.csv',baseline)
    write_json(output/'SUMMARY.json',dict(status='PASS_RAW_IQ_REBUILD_AND_REFIT',created_at=now(),**totals,
        max_C_error=max(r['C_max_error'] for r in baseline),max_score_error=max(r['score_max_error'] for r in baseline),
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
        raw_source_scope='S1/S4 published local IQ only; historical bank fixed, old C/H are comparison targets'))
    files = [p for p in output.rglob('*') if p.is_file()]
    write_json(output/'MANIFEST.json',dict(created_at=now(),entries=[dict(path=p.relative_to(output).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in files]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=HERE/'data/raw_rebuild_run01')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output)
