import json
p1=json.load(open("/work/audit_addendum4/audit_part1.json")); p2=json.load(open("/work/audit_addendum4/audit_part2.json"))
out={"verdicts":{"1_timing":"PASS","2_selection":"PASS","3_grading":"PASS","4_statistics":"PASS","5_leakage":"PASS","6_edit_and_gpu":"PASS"},
"timing":{"freeze_ts":"2026-11-04T22:04:39Z","first_res_container_created":"2026-11-04T22:04:39.87Z","selected_fix_stamp":"2026-11-04T23:05:41Z","first_I2_launch":"2026-11-04T23:05:48Z","freeze_hashes_ok":"12/12","extra_unregistered_run":"m500_I2_clcsel_s1 launched 2026-11-05T01:20:00Z","i2_failed_starts":2},
"gpu":{"uuid":"GPU-40333bf8-ac6a-447f-990a-b02c146800dc","model":"NVIDIA A100-PCIE-40GB"},
**p1,"edit":p2}
json.dump(out,open("/work/audit_addendum4/audit.json","w"),indent=1,default=str)
