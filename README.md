
# DRAMSimVS — Visual Studio Python Project

This is a ready-to-open **Visual Studio** project for the DRAM activation simulator.

## Files
- `DRAMSimVS.pyproj` — Visual Studio Python project file.
- `dram_sim.py` — the simulator.

## Open in Visual Studio
1. Open Visual Studio (Windows/macOS with Python workload installed).
2. `File` → `Open` → `Project/Solution`.
3. Select `DRAMSimVS.pyproj`.
4. In **Solution Explorer**, right-click the project → **Properties** → **Debug** to adjust `CommandLineArguments`.
5. Press **F5** to run.

### Example CommandLineArguments
```
--rows 32768 --trc 45ns --threshold 16000 --alert 64ms
```

## Notes
- You can switch between **Debug**/**Release** configurations to use the pre-filled arguments.
- If your Visual Studio uses a different Python version, set **Interpreter** in project properties.
