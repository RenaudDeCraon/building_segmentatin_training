"""
Data structure explorer to understand your downloaded dataset
"""
import os
from pathlib import Path
import pandas as pd

def explore_directory(base_path):
    """Explore the structure of your downloaded dataset"""
    base_path = Path(base_path)
    
    print(f"📁 Exploring: {base_path}")
    print("=" * 60)
    
    # Get all files and directories
    all_items = []
    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)
        level = len(root_path.relative_to(base_path).parts)
        
        # Print directory structure
        indent = "  " * level
        print(f"{indent}📁 {root_path.name}/")
        
        # Analyze files in this directory
        file_info = {}
        for file in files:
            file_path = root_path / file
            ext = file_path.suffix.lower()
            size_mb = file_path.stat().st_size / (1024 * 1024)
            
            if ext not in file_info:
                file_info[ext] = {'count': 0, 'total_size': 0, 'examples': []}
            
            file_info[ext]['count'] += 1
            file_info[ext]['total_size'] += size_mb
            
            if len(file_info[ext]['examples']) < 3:
                file_info[ext]['examples'].append(file)
            
            all_items.append({
                'path': str(file_path),
                'name': file,
                'extension': ext,
                'size_mb': size_mb,
                'directory': str(root_path)
            })
        
        # Print file summary for this directory
        for ext, info in file_info.items():
            examples = ', '.join(info['examples'][:3])
            if info['count'] > 3:
                examples += f" ... (+{info['count']-3} more)"
            
            print(f"{indent}  📄 {ext}: {info['count']} files, "
                  f"{info['total_size']:.1f} MB")
            print(f"{indent}     Examples: {examples}")
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    # Create summary DataFrame
    df = pd.DataFrame(all_items)
    if not df.empty:
        summary = df.groupby('extension').agg({
            'name': 'count',
            'size_mb': ['sum', 'mean']
        }).round(2)
        summary.columns = ['File Count', 'Total Size (MB)', 'Avg Size (MB)']
        print(summary)
        
        print(f"\n📈 Total files: {len(df)}")
        print(f"📈 Total size: {df['size_mb'].sum():.1f} MB")
        
        # Look for common patterns
        print("\n🔍 DETECTED PATTERNS:")
        
        # Check for images
        image_exts = ['.tif', '.tiff', '.jpg', '.jpeg', '.png']
        image_files = df[df['extension'].isin(image_exts)]
        if not image_files.empty:
            print(f"✅ Found {len(image_files)} image files")
        
        # Check for labels/masks
        label_patterns = ['label', 'mask', 'gt', 'truth', 'reference']
        label_files = df[df['name'].str.lower().str.contains('|'.join(label_patterns), na=False)]
        if not label_files.empty:
            print(f"✅ Found {len(label_files)} potential label files")
        
        # Check for geospatial data
        geo_exts = ['.geojson', '.shp', '.kml', '.gpkg']
        geo_files = df[df['extension'].isin(geo_exts)]
        if not geo_files.empty:
            print(f"✅ Found {len(geo_files)} geospatial files")
        
        # Check for RGB vs multispectral
        rgb_files = df[df['name'].str.lower().str.contains('rgb', na=False)]
        ms_files = df[df['name'].str.lower().str.contains('ms|multispectral', na=False)]
        
        if not rgb_files.empty:
            print(f"✅ Found {len(rgb_files)} RGB files")
        if not ms_files.empty:
            print(f"✅ Found {len(ms_files)} multispectral files")
    
    return df

# Usage
if __name__ == "__main__":
    # Change this to your downloaded data path
    data_path = input("Enter the path to your downloaded data: ").strip()
    
    if os.path.exists(data_path):
        df = explore_directory(data_path)
        
        print("\n" + "=" * 60)
        print("📋 NEXT STEPS RECOMMENDATIONS:")
        print("=" * 60)
        
        # Check what we found and give recommendations
        image_exts = ['.tif', '.tiff', '.jpg', '.jpeg', '.png']
        images = df[df['extension'].isin(image_exts)]
        
        if len(images) > 0:
            print("✅ You have image data!")
            print("🔧 Look for corresponding mask/label files")
            print("🔧 Check if images are paired with ground truth data")
        
        geo_exts = ['.geojson', '.shp', '.kml', '.gpkg']
        geo_files = df[df['extension'].isin(geo_exts)]
        
        if len(geo_files) > 0:
            print("✅ You have geospatial label data!")
            print("🔧 We'll need to convert these to pixel masks")
        
        if len(images) > 0 and len(geo_files) > 0:
            print("🎯 PERFECT! You have both images and labels")
            print("🚀 Ready to start data preparation!")
        
    else:
        print(f"❌ Path not found: {data_path}")
        print("Please check the path and try again.")