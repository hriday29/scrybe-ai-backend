"""
Script to find the correct Nifty symbol in Angel One's instrument list
This script directly downloads and searches the instrument list
"""
import requests
import json
from collections import defaultdict

INSTRUMENT_LIST_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

def find_nifty_symbols():
    """Search for Nifty-related instruments"""
    
    print("📥 Downloading instrument list from Angel One...")
    try:
        response = requests.get(INSTRUMENT_LIST_URL, timeout=30)
        response.raise_for_status()
        instruments = response.json()
        print(f"✅ Downloaded {len(instruments)} instruments\n")
    except Exception as e:
        print(f"❌ Error downloading instruments: {e}")
        return
    
    print("=" * 100)
    print("SEARCHING FOR NIFTY-RELATED INSTRUMENTS...")
    print("=" * 100)
    
    # Search for instruments with "NIFTY" in name or symbol
    nifty_instruments = []
    for inst in instruments:
        if not isinstance(inst, dict):
            continue
        
        name = inst.get('name', '').upper()
        symbol = inst.get('symbol', '').upper()
        
        if 'NIFTY' in name or 'NIFTY' in symbol:
            nifty_instruments.append(inst)
    
    print(f"\n✅ Found {len(nifty_instruments)} Nifty-related instruments\n")
    
    if not nifty_instruments:
        print("❌ No Nifty instruments found!")
        return
    
    # Group by instrument type
    grouped = defaultdict(list)
    for inst in nifty_instruments:
        inst_type = inst.get('instrumenttype', 'UNKNOWN')
        grouped[inst_type].append(inst)
    
    print("📊 Breakdown by Instrument Type:")
    print("-" * 100)
    for inst_type, items in sorted(grouped.items()):
        print(f"  {inst_type}: {len(items)} instruments")
    
    # Show indices first (most important for your use case)
    print("\n" + "=" * 100)
    print("🎯 NIFTY INDICES (Most Important)")
    print("=" * 100)
    
    indices = [inst for inst in nifty_instruments 
               if inst.get('instrumenttype') in ['', 'INDEX', 'AMXIDX']]
    
    if indices:
        print(f"\nFound {len(indices)} index instruments:\n")
        print(f"{'TOKEN':<12} {'SYMBOL':<20} {'NAME':<30} {'EXCHANGE':<10} {'TYPE':<10}")
        print("-" * 100)
        
        for inst in sorted(indices, key=lambda x: x.get('name', '')):
            token = inst.get('token', 'N/A')
            symbol = inst.get('symbol', 'N/A')
            name = inst.get('name', 'N/A')
            exchange = inst.get('exch_seg', 'N/A')
            inst_type = inst.get('instrumenttype', 'N/A')
            
            print(f"{token:<12} {symbol:<20} {name:<30} {exchange:<10} {inst_type:<10}")
    else:
        print("⚠️ No index instruments found")
    
    # Look specifically for Nifty 50
    print("\n" + "=" * 100)
    print("🔍 SEARCHING FOR MAIN NIFTY 50 INDEX...")
    print("=" * 100)
    
    nifty_50_candidates = []
    for inst in instruments:
        name = inst.get('name', '').upper()
        symbol = inst.get('symbol', '').upper()
        
        # Look for exact matches
        if any(pattern in name for pattern in ['NIFTY 50', 'NIFTY50', 'NIFTY-50']):
            nifty_50_candidates.append(inst)
        elif symbol in ['NIFTY', 'NIFTY50', 'NIFTY-50', 'NIFTY 50']:
            nifty_50_candidates.append(inst)
    
    if nifty_50_candidates:
        print(f"\n✅ Found {len(nifty_50_candidates)} Nifty 50 candidate(s):\n")
        print(f"{'TOKEN':<12} {'SYMBOL':<20} {'NAME':<30} {'EXCHANGE':<10} {'TYPE':<10}")
        print("-" * 100)
        
        for inst in nifty_50_candidates:
            token = inst.get('token', 'N/A')
            symbol = inst.get('symbol', 'N/A')
            name = inst.get('name', 'N/A')
            exchange = inst.get('exch_seg', 'N/A')
            inst_type = inst.get('instrumenttype', 'N/A')
            
            print(f"{token:<12} {symbol:<20} {name:<30} {exchange:<10} {inst_type:<10}")
            
        print("\n" + "=" * 100)
        print("💡 RECOMMENDATION")
        print("=" * 100)
        print("\nTo use in your code, try these exact values:")
        for inst in nifty_50_candidates[:3]:  # Show top 3
            print(f"\nOption {nifty_50_candidates.index(inst) + 1}:")
            print(f"  Symbol: '{inst.get('symbol', 'N/A')}'")
            print(f"  Name: '{inst.get('name', 'N/A')}'")
            print(f"  Exchange: '{inst.get('exch_seg', 'N/A')}'")
            print(f"  Token: '{inst.get('token', 'N/A')}'")
    else:
        print("⚠️ No exact Nifty 50 match found")
        print("\nShowing instruments with 'NIFTY' in name (top 10):")
        nifty_only = [inst for inst in nifty_instruments if 'NIFTY' in inst.get('name', '').upper()][:10]
        
        if nifty_only:
            print(f"\n{'TOKEN':<12} {'SYMBOL':<20} {'NAME':<30} {'EXCHANGE':<10}")
            print("-" * 100)
            for inst in nifty_only:
                token = inst.get('token', 'N/A')
                symbol = inst.get('symbol', 'N/A')
                name = inst.get('name', 'N/A')
                exchange = inst.get('exch_seg', 'N/A')
                print(f"{token:<12} {symbol:<20} {name:<30} {exchange:<10}")
    
    # Show sample of futures and options
    print("\n" + "=" * 100)
    print("📈 SAMPLE NIFTY FUTURES (First 5)")
    print("=" * 100)
    
    futures = [inst for inst in nifty_instruments 
               if inst.get('instrumenttype') in ['FUTIDX', 'FUTSTK']][:5]
    
    if futures:
        print(f"\n{'TOKEN':<12} {'SYMBOL':<25} {'NAME':<25} {'EXPIRY':<12}")
        print("-" * 100)
        for inst in futures:
            print(f"{inst.get('token', 'N/A'):<12} {inst.get('symbol', 'N/A'):<25} "
                  f"{inst.get('name', 'N/A'):<25} {inst.get('expiry', 'N/A'):<12}")

if __name__ == "__main__":
    find_nifty_symbols()