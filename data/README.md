# Data

Place the processed CSV files used by the SABLE execution workflow in this directory.

## Files

- `common_data.csv`: processed reference-day sequence records used for episodic task construction.
- `incident_data.csv`: processed incident-associated records and task metadata.

## Source

The data were prepared from the traffic and incident sources described in the accompanying manuscript and dataset record:

IEEE DataPort DOI: 10.21227/d6ht-c375

## Usage

The default configuration expects these files under the `data/` directory. The paper-oriented workflow selects `Volume`, subtracts the mean of `Data1`--`Data5` from the 25-point sequence, and constructs task keys from `DESCRIPTION + Week + tag`.

The software license in this repository applies to the source code. Data use remains subject to the terms of the cited data sources and dataset record.
