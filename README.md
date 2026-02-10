
# PRAC Denial of Service Simulator

This is a ready-to-use **Visual Studio Code** project for the PRAC (Per Row Activation Counting) Denial of Service simulator for DRAM ACTIVATEs with GLOBAL ALERT stalls.

## Features

- **Two simulation modes**:
  - `report`: Uses DRAM timings specific to the selected DRAM type (e.g., 'ddr5')
  - `explore`: DRAM timings are passed via command-line flags
- **Round-robin ACTIVATEs**: Activations cycle across N rows
- **GLOBAL ALERT stalls**: When a counter exceeds threshold, ALERT duration is consumed immediately (no ACTIVATEs to ANY row during ALERT)
- **Windowed RFM**: Configurable proactive RFM with randomized timing windows
- **Alert-based RFM**: Reactive RFMs triggered by threshold violations
- **Comprehensive metrics**: Tracks activations, alerts, RFMs, and per-row alert time
- **CSV output**: Parameter sweep-friendly output format
- **Flexible timing**: Support for ns, us (or µs), ms, s time units

## Setup

### Prerequisites
- **Python 3.7+**

### Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run the simulator (see Usage below)

## Usage

### Running the Simulator

The simulator supports two modes: `report` and `explore`.

**Report mode** (uses DRAM config file):
```bash
python dram_sim.py report --dram-type ddr5 --rows 8 --threshold 1000
```

**Explore mode** (all parameters via command line):
```bash
python dram_sim.py explore --rows 8 --trc 45ns --threshold 1000 --rfmabo 2 --trfcrfm 410ns --runtime 32ms
```

### Command Line Parameters

#### Common Parameters (both modes)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--rows` | Number of rows to operate on | `8` |
| `--threshold` | Counter threshold; ALERT raised when counter > threshold | `1000` |
| `--rfmfreqmin` | RFM window start time (use '0' to disable RFM) | `32us` |
| `--rfmfreqmax` | RFM window end time (must be >= rfmfreqmin, use '0' to disable RFM) | `48us` |
| `--csv` | CSV output format | (flag) |

#### Report Mode Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--dram-type` | DRAM type for loading protocol parameters from config | `ddr5` |

#### Explore Mode Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--trc` | tRC per ACTIVATE (e.g., '45ns', '3us', '64ms', '0.001s') | `45ns` |
| `--rfmabo` | RFM ABO multiplier; alert duration = rfmabo × trfcrfm | `4` |
| `--trfcrfm` | tRFC RFM time duration consumed when RFM is issued (use '0' for no time consumption) | `410ns` |
| `--runtime` | Total simulation runtime | `32ms` |

### Example Commands

**Report mode with DDR5 config:**
```bash
python dram_sim.py report --dram-type ddr5 --rows 4 --threshold 1000
```

**Report mode with windowed RFM:**
```bash
python dram_sim.py report --dram-type ddr5 --rows 8 --threshold 2000 --rfmfreqmin 24us --rfmfreqmax 36us
```

**Explore mode basic simulation:**
```bash
python dram_sim.py explore --rows 4 --trc 45ns --threshold 1000 --rfmabo 1 --trfcrfm 410ns --runtime 32ms
```

**Explore mode with windowed RFM:**
```bash
python dram_sim.py explore --rows 8 --trc 45ns --threshold 2000 --rfmabo 2 --trfcrfm 410ns --rfmfreqmin 24us --rfmfreqmax 36us --runtime 32ms
```

**CSV output for parameter sweeps:**
```bash
python dram_sim.py explore --rows 1 --trc 45ns --threshold 500 --rfmabo 4 --trfcrfm 410ns --runtime 32ms --csv
```

## Output Formats

### Standard Output
```
=== DRAM Activation Simulation Summary ===
Runtime:            32.000 ms
Total ACTIVATEs:    218364
Total RFMs issued:  2659
Per-row metrics:
   Row |  Activations | Alerts |   RFMs |   Alert Time
   ...
```

### CSV Output
```
rows,trc,threshold,rfmabo,rfmfreqmin,rfmfreqmax,trfcrfm,runtime,Row,Activations,Alerts,RFMs,AlertTime
8,45ns,1000,2,24us,36us,200ns,10ms,ALL,171651,0,6502,0.0
```

## Notes

- **Time Units**: Supports ns (nanoseconds), us (microseconds), ms (milliseconds), s (seconds)
- **RFM Types**: Both proactive (windowed) and reactive (alert-based) RFMs are counted
- **CSV Format**: Designed for easy parameter sweep analysis and data processing
