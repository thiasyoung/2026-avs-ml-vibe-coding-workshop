# Fixed IR evaluation reference

Keep these files in the workshop repository, including both Parquets and
`evaluation_contract.json`. They are approximately 8.6 MB together and allow
all participants to use the same evaluation task.

These are the fixed validation and test files used for every carpentry comparison.
Each contains 975 spectra on the original shared 500-point grid from 600.88402
to 3846 cm^-1. They retain the original per-spectrum min-max normalization.
The contract records both SHA-256 hashes, ordered labels and grid.

IR 01 changes **training data only** and excludes native observations matching
held-out molecular identities, CAS numbers or NIST IDs before carpentry.
IR 02 verifies the reference files and selected dataset's copies before training.
No experimental outlier rejection, resampling, padding or signal policy modifies
these held-out files. The same test examples, features, labels and ordering remain
in place, not merely the same random seed.

This preserves an existing benchmark, not a claim that its historical preparation
or split was ideal. It does not retroactively replace legacy validation/test
membership with the newer canonical-grouped splitting scheme. If you intentionally
change the reference grid, eligible measurements or reference processing, create
a new benchmark version and rerun every comparison; do not silently replace this
reference during an ordinary carpentry attempt. If files are missing, restore them
from the repository rather than initializing new splits in your workbook.
