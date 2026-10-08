# C2 hypothesis assessment

This is the predeclared analysis on the previously examined C1 sample. Paired
point differences below are percentage points; no equivalence tolerance,
significance criterion or training-seed confidence interval was declared.

## C2-H1 — dimensional reduction need not imply proportional utility loss

[measurement] MRL 512 versus native: weather/scene/timeofday balanced accuracy
changes -0.255/+0.593/+0.718 pp; P@10 changes -0.090/-0.200/-0.080 pp.
MRL 256: balanced changes -1.926/-0.135/+0.542 pp; P@10 changes
-0.510/-0.230/-0.320 pp. Nominal dimensions are reduced by one third/two thirds.

[interpretation] These point responses show much smaller changes in the reported
semantic endpoints than in nominal coordinate budget. They are consistent with
benign learned compression on these attributes. They do not establish formal
noninferiority/equivalence or a universally acceptable retention threshold.

## C2-H2 — matched dimensional budgets do not imply matched utility

[measurement] MRL-minus-PCA balanced accuracy at 512:
+7.189/+3.548/+1.932 pp (weather/scene/timeofday); at 256:
+2.054/+2.868/+1.474 pp; at 128: -7.304/-1.308/-3.942 pp.
The corresponding retrieval differences are +1.960/+0.430/-0.430 pp,
+1.260/+0.230/-0.930 pp, and +0.120/-1.410/-3.270 pp.

[interpretation] Equal D permits different semantic endpoints. Neither mechanism
wins every endpoint/dimension. Learned MRL compression is not uniquely benign:
PCA can also preserve utility. This is the declared centered, normalized PCA
baseline; centering and coordinate-wise probe standardization are not isolated
causal factors. Point advantages are conditional, without inferential verdicts.

## C2-H3 — aggressive learned compression may expose a boundary

[measurement] MRL 128 versus native balanced accuracy falls
7.857/2.828/2.214 pp; P@10 falls 1.500/1.630/2.050 pp. Compared with MRL 256,
BA falls 5.931/2.693/2.756 pp and P@10 falls 0.990/1.400/1.730 pp.
Raw timeofday accuracy is 0.919 at 128 versus 0.907 at 256, illustrating that
accuracy and balanced accuracy need not order dimensions identically.
PCA 128 versus native BA changes -0.554/-1.520/+1.728 pp and P@10 changes
-1.620/-0.220/+1.220 pp.

[interpretation] The prior expectation transfers descriptively to MRL balanced
accuracy and retrieval: 128 shows larger losses than 256/512 on these endpoints.
It is not a universal monotonic rule for raw accuracy or generic compression.
No post-hoc materiality threshold defines a sharp utility boundary.

## C2-H4 — rank is not a task-independent utility score

[measurement] MRL raw RankMe decreases 271.900→252.012→150.567→72.184,
while scene/timeofday BA initially rises slightly at 512 and timeofday BA also
exceeds native at 256. PCA raw RankMe decreases 349.317→204.963→109.659 and
participation ratio decreases 34.567→30.016→23.395 as D decreases; yet weather
BA increases 0.5379→0.5725→0.6068, and timeofday P@10 increases
0.8228→0.8254→0.8315. These PCA dimension-specific trends are exploratory
mechanism/task observations within the predeclared rank-versus-utility comparison.

[interpretation] Raw rank cannot supply a monotonic utility ordering across
these constructions and tasks. Capacity-normalized values describe occupied
nominal capacity, not semantic quality. Native/MRL retain a large common mean;
centered PCA removes it, affecting raw RankMe and cosine independently of
semantic endpoints. Do not call any valid lower-dimensional representation
collapsed on the basis of D or rank alone.

## Fit limitations and open questions

[measurement] All 21 final standardized fits converge; 18 select C=.01 at the
lower boundary. Of 168 inner candidates, 19 reach max_iter=5000 and do not
converge. Candidate warnings, scores, support and final diagnostics are retained.
Native C2 probe accuracy/BA exactly reproduces C1. Completed resume performs
zero endpoint calls and preserves endpoint/effect/PCA-parameter hashes.

[interpretation] Convergence does not prove an interior regularization optimum.
No C grid or solver was retuned after inspecting results. Especially close
point differences should not become population or encoder-family claims.

[open] Results remain conditional on one pretrained encoder and a previously
examined sample. Repository training-seed intervals are not estimable; rare
BDD labels have insufficient eligible support. The causes of PCA/MRL differences
(centering, normalization, selected regularization, coordinate geometry) are
not separately identified. No random projection was added: a future control
requires a specific new question. C3/C4 remain deferred.
