# Original paired-source qualification

[ours] This namespace records the original COCO2017 validation release, not an
enriched repackaging. Acquisition, inventory, content-inspection, sample and
rendered-input ledgers were frozen at `2eee895` before any encoder inference.
All source IDs and original UTF-8 caption text are preserved. Byte continuity
and dataset-source authentication are separate checks.

[ours] Completed checks: certificate-verified original publisher-named S3 bucket
transport; saved response headers and publisher release/schema/terms documents;
local archive and annotation SHA256; annotation archive CRC; all 5,000 original
image-member CRC/RGB decodes/dimensions; caption/instance image dictionaries,
unique IDs, foreign keys, filename IDs and licence references; raw-byte,
dimension-bound decoded-pixel and recovered Flickr-photo grouping; deterministic
1,000-group selection and first-five annotation IDs; all 1,000 selected thumbnail
and caption bundles inspected in 50 contact sheets. The publisher alias had a TLS
hostname mismatch; the documented pre-freeze amendment uses the same S3 bucket's
certificate-valid path-style address. Certificate validation was never bypassed.
No independently publisher-signed archive checksum was verified.

[ours] Exact raw/pixel/photo grouping yields 5,000 distinct release groups. The
source has 25,014 caption annotations. Selected captions contain 18 repeated string
types and 24 rows beyond their first occurrence, with 17 repeated types crossing
parents. These retain separate original annotation IDs and parent relevance;
identical strings do not manufacture additional judged positives.

[ours] Inspection was assistant thumbnail/caption review, not independent human
relevance adjudication or full-resolution forensic review. Some long displayed
captions wrap beyond the contact-sheet panel; complete original strings remain
in the frozen inputs. Original illustrations (e.g. image 6614), edited photographs
(e.g. 344268) and factual caption errors (e.g. 42296) were recorded before freeze
and retained. Original non-descriptive annotations also remain: caption 433639
for image 508586 says no image is present, and caption 57209 for image 554838
states uncertainty about the content. These examples are source annotations,
not extraction placeholders or fallback-generated rows; no post-result filtering
or independent caption adjudication was performed.

[interpretation] Remaining limits: unknown encoder pretraining overlap; incomplete
cross-parent relevance; resized/near/event duplicates not detected by exact
hash/photo grouping; correlated scenes/photographers; original annotation errors;
rights retained by image owners under the recorded release licences. Authentication
of a release does not certify every caption or prove unedited photography.
Sampling is independently selected evaluation data, with the same encoder.

[interpretation] Nothing here retroactively authenticates the enriched BDD data.
The current BDD analysis remains the 983-row post-audit sensitivity with verified
fragment provenance and its stated limits. All historical campaigns are preserved;
the exploratory pilot's six separately unresolved appended-fragment IDs remain
qualified. This study does not need a claim drawn from that pilot.
