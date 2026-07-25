import os
import re
import json

base_dir = r"c:\Users\sayak\Downloads\coding shits\physics\MAIN\13CD_cross_section_data-20260707T150327Z-3-001\13CD_cross_section_data\13CD_cross_section_20260413\DAQ"

if not os.path.exists(base_dir):
    base_dir = r"..\13CD_cross_section_data-20260707T150327Z-3-001\13CD_cross_section_data\13CD_cross_section_20260413\DAQ"

folders = [f for f in os.listdir(base_dir) if f.startswith("2026") and os.path.isdir(os.path.join(base_dir, f))]

ch1_stats = {"input": 0, "pileup": 0, "output": 0, "live_time": 0.0, "real_time": 0.0}
ch2_stats = {"input": 0, "pileup": 0, "output": 0, "live_time": 0.0, "real_time": 0.0}

for fld in folders:
    info_p = os.path.join(base_dir, fld, f"{fld}_info.txt")
    if not os.path.exists(info_p):
        continue
    txt = open(info_p, encoding="utf-8", errors="replace").read()
    
    # Extract CH1 block
    m1 = re.search(r"CH1@.*?(?=CH2@|CH3@|\Z)", txt, re.S)
    if m1:
        blk1 = m1.group(0)
        lt1 = re.search(r"Live time\s*=\s*(\d+):(\d+):([\d.]+)", blk1)
        rt1 = re.search(r"Real time\s*=\s*(\d+):(\d+):([\d.]+)", blk1)
        inp1 = re.search(r"Input counts\s*=\s*(\d+)", blk1)
        pu1 = re.search(r"Pile up counts\s*=\s*(\d+)", blk1)
        out1 = re.search(r"Output counts\s*=\s*(\d+)", blk1)
        
        if lt1 and inp1:
            ch1_stats["live_time"] += int(lt1.group(1))*3600 + int(lt1.group(2))*60 + float(lt1.group(3))
            ch1_stats["real_time"] += int(rt1.group(1))*3600 + int(rt1.group(2))*60 + float(rt1.group(3))
            ch1_stats["input"] += int(inp1.group(1))
            ch1_stats["pileup"] += int(pu1.group(1))
            ch1_stats["output"] += int(out1.group(1))
            
    # Extract CH2 block
    m2 = re.search(r"CH2@.*?(?=CH3@|\Z)", txt, re.S)
    if m2:
        blk2 = m2.group(0)
        lt2 = re.search(r"Live time\s*=\s*(\d+):(\d+):([\d.]+)", blk2)
        rt2 = re.search(r"Real time\s*=\s*(\d+):(\d+):([\d.]+)", blk2)
        inp2 = re.search(r"Input counts\s*=\s*(\d+)", blk2)
        pu2 = re.search(r"Pile up counts\s*=\s*(\d+)", blk2)
        out2 = re.search(r"Output counts\s*=\s*(\d+)", blk2)
        
        if lt2 and inp2:
            ch2_stats["live_time"] += int(lt2.group(1))*3600 + int(lt2.group(2))*60 + float(lt2.group(3))
            ch2_stats["real_time"] += int(rt2.group(1))*3600 + int(rt2.group(2))*60 + float(rt2.group(3))
            ch2_stats["input"] += int(inp2.group(1))
            ch2_stats["pileup"] += int(pu2.group(1))
            ch2_stats["output"] += int(out2.group(1))

print("=== DAQ INFO HARDWARE RECORDINGS FOR ALL RUNS ===")
print("CHANNEL 1 (CH1):")
print(f"  Live time : {ch1_stats['live_time']:.1f} s ({ch1_stats['live_time']/3600:.2f} h)")
print(f"  Real time : {ch1_stats['real_time']:.1f} s ({ch1_stats['real_time']/3600:.2f} h)")
print(f"  Input counts : {ch1_stats['input']:,d}")
print(f"  Hardware Pile-up rejected: {ch1_stats['pileup']:,d}")
print(f"  Output counts: {ch1_stats['output']:,d}")
print()
print("CHANNEL 2 (CH2):")
print(f"  Live time : {ch2_stats['live_time']:.1f} s ({ch2_stats['live_time']/3600:.2f} h)")
print(f"  Real time : {ch2_stats['real_time']:.1f} s ({ch2_stats['real_time']/3600:.2f} h)")
print(f"  Input counts : {ch2_stats['input']:,d}")
print(f"  Hardware Pile-up rejected: {ch2_stats['pileup']:,d}")
print(f"  Output counts: {ch2_stats['output']:,d}")
