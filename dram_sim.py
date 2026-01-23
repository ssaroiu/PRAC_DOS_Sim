#!/usr/bin/env python3
"""
DRAM activation simulator for a configurable runtime (default 128 ms).

Behavior:
- Round-robin ACTIVATEs across N rows.
- Each ACTIVATE consumes tRC time.
- Each ACTIVATE increments a per-row counter.
- If a counter strictly exceeds the threshold, raise an ALERT:
    * ALERT duration is consumed immediately (GLOBAL STALL).
    * No ACTIVATEs (to ANY row) occur while ALERT time is being consumed.
- Record total time each row spent in ALERT state.
- Supports time inputs with units: ns, us (or µs), ms, s.

Inputs:
- --rows           Number of rows to operate on.
- --trc            tRC per ACTIVATE (e.g., '45ns', '3.2us', '64ms', '0.001s').
- --threshold      Counter threshold; ALERT raised when counter > threshold.
- --rfmabo         Number of RFMs issued in response to ABO.
- --rfmfreqmin     RFM window start time (e.g., '32us', '64us'). Use '0' to disable RFM.
- --rfmfreqmax     RFM window end time (e.g., '48us', '80us'). Must be >= rfmfreqmin. Use '0' to disable RFM.
- --trfcrfm        tRFC RFM time duration consumed when RFM is issued (e.g., '100ns', '1us'). Use '0' for no time consumption.
- --runtime        Total simulation runtime (default 128 ms).

Notes:
- ALERT time is a GLOBAL STALL: while an alert is active, no activates occur.
- ALERT starts immediately after the ACTIVATE that triggered it.
- If remaining runtime is shorter than the alert duration, only the remaining time is consumed and counted.
"""

import argparse
import sys
import random
from typing import List


def parse_time_to_seconds(s: str) -> float:
    """
    Parse a time string to seconds. Accepts:
        - Plain numbers (assumed seconds): e.g., "0.128"
        - With units: "ns", "us" (or "µs"), "ms", "s"
          Examples: "45ns", "3.2us", "64ms", "0.128s"
    """
    s = s.strip().lower().replace("µs", "us")
    unit = None
    for candidate in ("ns", "us", "ms", "s"):
        if s.endswith(candidate):
            unit = candidate
            numeric = s[: -len(candidate)].strip()
            break
    if unit is None:
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"Invalid time format: '{s}'")
    try:
        value = float(numeric)
    except ValueError:
        raise ValueError(f"Invalid numeric time: '{s}'")

    return {
        "s": value,
        "ms": value * 1e-3,
        "us": value * 1e-6,
        "ns": value * 1e-9,
    }[unit]


def human_time(seconds: float) -> str:
    """Format a time in seconds using a friendly unit selection."""
    abs_s = abs(seconds)
    if abs_s >= 1.0:
        return f"{seconds:.6f} s"
    elif abs_s >= 1e-3:
        return f"{seconds * 1e3:.3f} ms"
    elif abs_s >= 1e-6:
        return f"{seconds * 1e6:.3f} us"
    else:
        return f"{seconds * 1e9:.3f} ns"


class DRAMSimulator:
    def __init__(
        self,
        rows: int,
        trc_s: float,
        threshold: int,
        rfmabo: int,
        runtime_s: float = 0.128,
        rfm_freq_min_s: float = 0.0,
        rfm_freq_max_s: float = 0.0,
        trfcrfm_s: float = 0.0,
        # Original string arguments for CSV output
        trc_str: str = "",
        rfmfreqmin_str: str = "",
        rfmfreqmax_str: str = "",
        trfcrfm_str: str = "",
        runtime_str: str = "",
    ):
        # Validate inputs
        if rows <= 0:
            raise ValueError("rows must be > 0")
        if trc_s <= 0:
            raise ValueError("tRC must be > 0")
        if threshold < 0:
            raise ValueError("threshold must be >= 0")
        if rfmabo < 0:
            raise ValueError("rfmabo must be >= 0")
        if runtime_s <= 0:
            raise ValueError("runtime must be > 0")
        if rfm_freq_min_s < 0:
            raise ValueError("RFM frequency min must be >= 0")
        if rfm_freq_max_s < 0:
            raise ValueError("RFM frequency max must be >= 0")
        if rfm_freq_min_s > 0 and rfm_freq_max_s > 0 and rfm_freq_max_s < rfm_freq_min_s:
            raise ValueError("RFM frequency max must be >= RFM frequency min")
        if trfcrfm_s < 0:
            raise ValueError("tRFC RFM time must be >= 0")

        # Parameters
        self.rows = rows
        self.trc_s = trc_s
        self.threshold = threshold
        self.rfmabo = rfmabo
        self.alert_duration_s = rfmabo * trfcrfm_s  # Calculate alert duration
        self.runtime_s = runtime_s
        self.rfm_freq_min_s = rfm_freq_min_s
        self.rfm_freq_max_s = rfm_freq_max_s
        self.trfcrfm_s = trfcrfm_s
        
        # Store original string arguments for CSV output
        self.trc_str = trc_str
        self.rfmfreqmin_str = rfmfreqmin_str
        self.rfmfreqmax_str = rfmfreqmax_str
        self.trfcrfm_str = trfcrfm_str
        self.runtime_str = runtime_str

        # State
        self.time_s: float = 0.0
        self.row_index: int = 0
        self.counters: List[int] = [0] * rows  # Threshold checking counters (reset on alert)
        self.total_activations_per_row: List[int] = [0] * rows  # Total activations per row (never reset)
        self.alerts_issued: List[int] = [0] * rows
        self.total_alert_time_s: List[float] = [0.0] * rows
        self.total_activations: int = 0
        
        # RFM state - windowed approach
        self.rfm_enabled = rfm_freq_min_s > 0 and rfm_freq_max_s > 0
        if self.rfm_enabled:
            self.next_rfm_window_start_s: float = rfm_freq_min_s
            self.next_rfm_window_end_s: float = rfm_freq_max_s
            self._schedule_next_rfm_in_window()  # Schedule first RFM within the first window
        else:
            self.next_rfm_window_start_s: float = float('inf')
            self.next_rfm_window_end_s: float = float('inf')
            self.next_rfm_time_s: float = float('inf')
        self.rfm_issued: List[int] = [0] * rows  # Track RFMs issued per row
        self.total_rfms: int = 0
        self.total_rfm_time_s: float = 0.0

    def _schedule_next_rfm_in_window(self):
        """Schedule the next RFM at a random time within the current window."""
        if self.rfm_enabled:
            window_duration = self.next_rfm_window_end_s - self.next_rfm_window_start_s
            if window_duration > 0:
                # Random time within the window
                random_offset = random.uniform(0, window_duration)
                self.next_rfm_time_s = self.next_rfm_window_start_s + random_offset
            else:
                # No window duration, schedule at window start
                self.next_rfm_time_s = self.next_rfm_window_start_s
        else:
            self.next_rfm_time_s = float('inf')

    def run(self):
        """
        Run the simulation until the runtime elapses.
        Step:
          - If there's enough time for an ACTIVATE (tRC), perform it.
          - If it triggers an ALERT, consume alert duration immediately (GLOBAL STALL).
        """
        while True:
            # Check if it's time for RFM before next activation
            if self.time_s >= self.next_rfm_time_s and self.rfm_enabled:
                # Check if we're still within the window
                if self.time_s <= self.next_rfm_window_end_s:
                    self._issue_rfm()
                
                # Check if window has expired, schedule next window
                if self.time_s >= self.next_rfm_window_end_s:
                    # Move to next window
                    self.next_rfm_window_start_s += self.rfm_freq_min_s
                    self.next_rfm_window_end_s = self.next_rfm_window_start_s + (self.rfm_freq_max_s - self.rfm_freq_min_s)
                    self._schedule_next_rfm_in_window()
                
            # Can we start an ACTIVATE within the runtime?
            if self.time_s + self.trc_s > self.runtime_s:
                break

            # ACTIVATE current row
            row = self.row_index
            self.counters[row] += 1  # Threshold checking counter
            self.total_activations_per_row[row] += 1  # Total activations counter
            self.total_activations += 1
            self.time_s += self.trc_s  # activation time consumed

            # Check threshold and possibly raise alert (GLOBAL STALL)
            if self.counters[row] > self.threshold and self.alert_duration_s > 0.0:
                remaining = self.runtime_s - self.time_s
                if remaining > 0.0:
                    consume = min(self.alert_duration_s, remaining)
                    self.alerts_issued[row] += 1
                    self.total_alert_time_s[row] += consume
                    self.time_s += consume
                
                # Issue rfmabo number of RFMs targeting highest counter rows
                self._issue_alert_rfms()

            # Next row (round robin)
            self.row_index = (self.row_index + 1) % self.rows
            
    def _issue_alert_rfms(self):
        """Issue rfmabo number of RFMs targeting rows with highest counters during alert."""
        # Get list of (counter_value, row_index) pairs for rows with non-zero counters
        rows_with_counts = [(self.counters[r], r) for r in range(self.rows) if self.counters[r] > 0]
        
        # Sort by counter value in descending order (highest first)
        rows_with_counts.sort(reverse=True, key=lambda x: x[0])
        
        # Issue up to rfmabo RFMs to the rows with highest counters
        rfms_to_issue = min(self.rfmabo, len(rows_with_counts))
        for i in range(rfms_to_issue):
            target_row = rows_with_counts[i][1]
            self.counters[target_row] = 0
            self.rfm_issued[target_row] += 1
            self.total_rfms += 1
            
    def _issue_rfm(self):
        """Issue RFM to the row closest to exceeding threshold."""
        # Find row with highest counter value (closest to threshold)
        max_counter = max(self.counters)
        if max_counter > 0:  # Only issue RFM if there are activations to reset
            # Find the first row with the maximum counter value
            target_row = self.counters.index(max_counter)
            self.counters[target_row] = 0
            self.rfm_issued[target_row] += 1
            self.total_rfms += 1
            
            # Consume RFM time if specified and runtime allows
            if self.trfcrfm_s > 0:
                remaining = self.runtime_s - self.time_s
                if remaining > 0:
                    consume = min(self.trfcrfm_s, remaining)
                    self.total_rfm_time_s += consume
                    self.time_s += consume
        
        # Note: Next RFM is scheduled in the run loop when window expires

    def summary(self) -> str:
        """Build a human-readable summary of the simulation results."""
        used_time = self.time_s
        idle_time = max(0.0, self.runtime_s - used_time)
        total_alert = sum(self.total_alert_time_s)

        lines = []
        lines.append("=== DRAM Activation Simulation Summary ===")
        lines.append(f"Runtime:            {human_time(self.runtime_s)}")
        lines.append(f"tRC per activate:   {human_time(self.trc_s)}")
        lines.append(f"Rows:               {self.rows}")
        lines.append(f"Threshold (>):      {self.threshold}")
        lines.append(f"RFM ABO:            {self.rfmabo}")
        if self.trfcrfm_s > 0:
            lines.append(f"Alert duration:     {human_time(self.alert_duration_s)} (RFM ABO × tRFC RFM)")
        else:
            lines.append(f"Alert duration:     {human_time(self.alert_duration_s)}")
        lines.append("")
        lines.append(f"Total ACTIVATEs:    {self.total_activations}")
        lines.append(f"Used time:          {human_time(used_time)}")
        lines.append(f"Idle time:          {human_time(idle_time)}")
        lines.append(f"Total alert time:   {human_time(total_alert)}")
        if self.rfm_enabled:
            window_duration = self.rfm_freq_max_s - self.rfm_freq_min_s
            lines.append(f"RFM window start:   {human_time(self.rfm_freq_min_s)}")
            lines.append(f"RFM window end:     {human_time(self.rfm_freq_max_s)}")
            lines.append(f"RFM window duration:{human_time(window_duration)}")
            lines.append(f"Total RFMs issued:  {self.total_rfms}")
            if self.trfcrfm_s > 0:
                lines.append(f"tRFC RFM time:      {human_time(self.trfcrfm_s)}")
                lines.append(f"Total RFM time:     {human_time(self.total_rfm_time_s)}")
        lines.append("")
        lines.append("Per-row metrics:")
        if self.rfm_enabled:
            lines.append(f"{'Row':>6} | {'Activations':>12} | {'Alerts':>6} | {'RFMs':>6} | {'Alert Time':>12}")
            lines.append("-" * 58)
        else:
            lines.append(f"{'Row':>6} | {'Activations':>12} | {'Alerts':>6} | {'Alert Time':>12}")
            lines.append("-" * 46)
        for r in range(self.rows):
            if self.rfm_enabled:
                lines.append(
                    f"{r:6d} | {self.total_activations_per_row[r]:12d} | {self.alerts_issued[r]:6d} | {self.rfm_issued[r]:6d} | {human_time(self.total_alert_time_s[r]):>12}"
                )
            else:
                lines.append(
                    f"{r:6d} | {self.total_activations_per_row[r]:12d} | {self.alerts_issued[r]:6d} | {human_time(self.total_alert_time_s[r]):>12}"
                )
        return "\n".join(lines)

    def csv_output(self) -> str:
        """Output metrics in CSV format: rows,trc,threshold,rfmabo,rfmfreqmin,rfmfreqmax,trfcrfm,runtime,Row,Activations,Alerts,RFMs,AlertTime"""
        # Input parameters first
        input_params = f"{self.rows},{self.trc_str},{self.threshold},{self.rfmabo},{self.rfmfreqmin_str},{self.rfmfreqmax_str},{self.trfcrfm_str},{self.runtime_str}"
        
        if self.rows == 1:
            # Single row - output as before with row number
            if self.rfm_enabled:
                metrics = f"0,{self.total_activations_per_row[0]},{self.alerts_issued[0]},{self.rfm_issued[0]},{self.total_alert_time_s[0] / 1e6}"
            else:
                metrics = f"0,{self.total_activations_per_row[0]},{self.alerts_issued[0]},0,{self.total_alert_time_s[0] / 1e6}"
        else:
            # Multiple rows - output summed totals with "ALL" as row identifier
            total_activations = sum(self.total_activations_per_row)
            total_alerts = sum(self.alerts_issued)
            total_alert_time_ms = sum(self.total_alert_time_s) / 1e6
            if self.rfm_enabled:
                total_rfms = sum(self.rfm_issued)
                metrics = f"ALL,{total_activations},{total_alerts},{total_rfms},{total_alert_time_ms}"
            else:
                metrics = f"ALL,{total_activations},{total_alerts},0,{total_alert_time_ms}"
        
        return f"{input_params},{metrics}"


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Simulate DRAM ACTIVATEs with GLOBAL ALERT stalls.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--rows", type=int, required=True, help="Number of rows to operate on.")
    p.add_argument("--trc", type=str, required=True, help="tRC per ACTIVATE (e.g., '45ns', '3us', '64ms', '0.001s').")
    p.add_argument(
        "--threshold", type=int, required=True,
        help="Counter threshold; ALERT raised when counter strictly exceeds this value."
    )
    p.add_argument(
        "--rfmabo", type=int, required=True,
        help="RFM ABO multiplier; alert duration = rfmabo × trfcrfm."
    )
    p.add_argument("--runtime", type=str, default="128ms", help="Total simulation runtime. Default is 128ms.")
    p.add_argument(
        "--rfmfreqmin", type=str, default="0",
        help="RFM (Row Fresh Management) window start time (e.g., '32us', '64us'). Use '0' to disable RFM. Default is 0 (disabled)."
    )
    p.add_argument(
        "--rfmfreqmax", type=str, default="0",
        help="RFM (Row Fresh Management) window end time (e.g., '48us', '80us'). Must be >= rfmfreqmin. Default is 0 (disabled)."
    )
    p.add_argument(
        "--trfcrfm", type=str, default="0",
        help="tRFC RFM time duration consumed when RFM is issued (e.g., '100ns', '1us'). Use '0' for no time consumption. Default is 0."
    )
    p.add_argument(
        "--csv", action="store_true",
        help="Output results in CSV format: Row,Activations,Alerts,RFMs,AlertTime"
    )
    return p


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        trc_s = parse_time_to_seconds(args.trc)
        runtime_s = parse_time_to_seconds(args.runtime)
        rfm_freq_min_s = parse_time_to_seconds(args.rfmfreqmin)
        rfm_freq_max_s = parse_time_to_seconds(args.rfmfreqmax)
        trfcrfm_s = parse_time_to_seconds(args.trfcrfm)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    # Validate RFM frequency range
    if rfm_freq_min_s > 0 and rfm_freq_max_s > 0 and rfm_freq_max_s < rfm_freq_min_s:
        print(f"Error: rfmfreqmax ({args.rfmfreqmax}) must be >= rfmfreqmin ({args.rfmfreqmin})", file=sys.stderr)
        return 2

    sim = DRAMSimulator(
        rows=args.rows,
        trc_s=trc_s,
        threshold=args.threshold,
        rfmabo=args.rfmabo,
        runtime_s=runtime_s,
        rfm_freq_min_s=rfm_freq_min_s,
        rfm_freq_max_s=rfm_freq_max_s,
        trfcrfm_s=trfcrfm_s,
        # Pass original string arguments for CSV output
        trc_str=args.trc,
        rfmfreqmin_str=args.rfmfreqmin,
        rfmfreqmax_str=args.rfmfreqmax,
        trfcrfm_str=args.trfcrfm,
        runtime_str=args.runtime,
    )
    sim.run()
    if args.csv:
        print(sim.csv_output())
    else:
        print(sim.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
