"""Attach explicit matrices to existing fixed-family regression rows.

Post-run representation/verification only: no new family, margin, grid,
sample selection, or rate estimate is introduced.
"""
import json
import numpy as np
from common import HERE, sha, now, write_json

if __name__ == '__main__':
    source = HERE/'results/THEORY_REGRESSIONS_EXTENDED_RUN02.json'
    rows = [r for r in json.loads(source.read_text())['rows']
            if r['test'] == 'fixed_matrix_intercept_only_rate']
    families = []
    z = np.diag([1., 0., 0.])
    for row in rows:
        name = row['family']
        h = np.zeros((3, 3))
        if name == 'cross':
            h[0, 2] = h[2, 0] = -1.
        elif name == 'linear':
            h[2, 2] = -1.
        elif name == 'mixed':
            h[1, 2] = h[2, 1] = 1.
        else:
            h[1, 1] = -1.
        checks = []
        for margin, radius in zip(row['margins'], row['first_failure_radii']):
            if name in ['cross', 'linear']:
                s, t = radius/np.sqrt(2), 0.
                psi = np.array([np.sqrt(1-s), 0., np.sqrt(s)])
            elif name == 'mixed':
                s, t = radius/np.sqrt(6), radius*radius/3
                psi = np.array([np.sqrt(1-s-t), np.sqrt(t), -np.sqrt(s)])
            else:
                s, t = 0., radius*radius/2
                psi = np.array([np.sqrt(1-t), np.sqrt(t), 0.])
            obs = np.zeros((3, 3))
            obs[:2, :2] = np.outer(psi[:2], psi[:2])
            obs[2, 2] = psi[2]**2
            score = float(margin+psi@h@psi)
            distance = float(np.linalg.norm(obs-z))
            assert abs(score) < 1e-15 and abs(distance-radius) < 1e-15
            assert abs(np.linalg.norm(psi)-1) < 1e-15
            checks.append(dict(margin=margin, psi=psi.tolist(),
                               pair_score=score, observation_distance=distance))
        families.append(dict(family=name, difference_matrix=h.tolist(),
            candidate_score='m + psi* H psi', competitor_score='0',
            candidate_intercept='m (only varied parameter)',
            base_state=[1., 0., 0.], K=1, T=3, q=2, d=1, checks=checks))
    write_json(HERE/'provenance/GAUSSIAN_FAMILY_COORDINATES.json',
        dict(created_at=now(), issuance='post-run explicit coordinates for existing four regression families',
             prior_results_sha256=sha(source), runner_sha256=sha(__file__),
             families=families, checks=68, status='PASS',
             scope='Witness coordinates verify saved formulas; completeness of endpoint laws is proved in the paper.'))
    print('PASS: 4 fixed matrices, 68 existing-margin witness coordinates')
