#!/usr/bin/env bash
# Download the Enrichr libraries used by omxpy into ref/enrich_db/human/
# Re-running skips files that already exist; add --force to the python call to refresh.

cd "$(dirname "$0")" || exit 1

DATABASES=(
  GO_Biological_Process_2023
  GO_Biological_Process_2025
  GO_Biological_Process_2026
  GO_Cellular_Component_2023
  GO_Cellular_Component_2025
  GO_Cellular_Component_2026
  GO_Molecular_Function_2023
  GO_Molecular_Function_2025
  GO_Molecular_Function_2026
  KEGG_2021_Human
  KEGG_2026
)

failed=()
for db in "${DATABASES[@]}"; do
  echo "== $db"
  python db_download_gp.py --database "$db" --organism human || failed+=("$db")
done

echo
if [ ${#failed[@]} -eq 0 ]; then
  echo "All ${#DATABASES[@]} libraries are in $(pwd)/human"
else
  echo "FAILED: ${failed[*]}  (re-run this script to retry only these)"
  exit 1
fi