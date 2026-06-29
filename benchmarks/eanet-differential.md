# eanet differential — independent-oracle check

The committed golden (`tests/fixtures/runes/eanet_sample.pcap` →
`tests/fixtures/runes/golden/eanet_parity.json`, pinned by
`tests/cross_arch_bit_identity.rs::parity_eanet_incident_pcap`) proves eanet
recovers a **known, injected** scanner + exfil byte-identically across arches.
But a synthetic fixture whose generator and parser share assumptions only
validates self-agreement (see the `oracle-shares-spec-errors` lesson). The real
check is a differential against an **independent** tool on a **wild** capture.

## Procedure (manual, non-CI — tshark is not a build dependency)

Use a real labelled capture with documented ground truth. The
[Stratosphere CTU-13](https://mcfp.felk.cvut.cz/publicDatasets/) botnet captures
work well — each scenario's `README.html` names the infected host(s).

```sh
# Real Neris botnet capture; README documents infected host = 147.32.84.165.
curl -O https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/botnet-capture-20110810-neris.pcap

# Independent oracle: tshark's IP-conversation table.
tshark -r botnet-capture-20110810-neris.pcap -q -z conv,ip | head

# eanet's view.
olorin rune eanet botnet-capture-20110810-neris.pcap
```

**Pass criteria:** the host eanet ranks #1 by source fan-out (and flags as a
scan) is the host the dataset README documents as infected, and it dominates
tshark's conversation table. They must agree without eanet being told the
ground truth.

## Result (validated 2026-06-29)

On CTU-Malware-Capture-Botnet-42 (Neris) and -50 (10-bot Neris), eanet's generic
fan-out / byte-volume metrics ranked the documented infected hosts
(`147.32.84.165`; the full `147.32.84.x` cluster) #1, matching tshark's
conversation table — with no knowledge of the labels. On the 1.1 GB capture the
deterministic scan was ~64× faster than `tshark -z conv,ip`. See memory
`eanet-pcap-wedge-validated` for the full run.
