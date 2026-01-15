import unittest
import sys
import shutil
from pathlib import Path
import geopandas as gpd
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src import demo_data
from src.violation_detection import ViolationDetector

class TestDemoMode(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test data
        self.test_dir = project_root / "data" / "test_boundaries"
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True)
        
    def tearDown(self):
        # clean up
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_end_to_end_demo_flow(self):
        # 1. Generate Synthetic Data
        print("\nTesting Data Generation...")
        generator = demo_data.DemoDataGenerator(str(self.test_dir))
        aoi_path = generator.generate_aoi()
        no_go_path = generator.generate_no_go_zones()
        mine_path = generator.generate_mine_boundary()
        
        self.assertTrue(Path(aoi_path).exists())
        self.assertTrue(Path(no_go_path).exists())
        self.assertTrue(Path(mine_path).exists())
        
        # 2. Simulate Excavation Growth
        print("Testing Excavation Simulation...")
        start_date = datetime(2025, 1, 1)
        results = generator.simulate_excavation_growth(start_date, days=30)
        
        self.assertEqual(len(results), 30)
        first_date = list(results.keys())[0]
        self.assertIsInstance(results[first_date], gpd.GeoDataFrame)
        
        # Check growth (last day area > first day area)
        last_date = list(results.keys())[-1]
        area_start = results[first_date].area.sum()
        area_end = results[last_date].area.sum()
        self.assertGreater(area_end, area_start)
        
        # 3. Detect Violations
        print("Testing Violation Detection...")
        detector = ViolationDetector(no_go_path)
        
        # Prepare inputs
        dates = sorted(list(results.keys()))
        excavation_data_list = [results[d] for d in dates]
        transforms = [None] * len(dates)
        crs_list = [data.crs for data in excavation_data_list]
        
        # Run detection
        violation_df, alerts = detector.track_violations_over_time(
            excavation_masks=excavation_data_list,
            dates=dates,
            transforms=transforms,
            crs_list=crs_list
        )
        
        print(f"Generated {len(alerts)} alerts")
        
        # We expect some alerts because the simulation grows into no-go zones
        self.assertTrue(len(alerts) > 0)
        self.assertIn('violation_type', alerts[0])
        self.assertIn('severity', alerts[0])
        
        # Verify DataFrame structure
        self.assertFalse(violation_df.empty)
        self.assertIn('total_violation_area_ha', violation_df.columns)

if __name__ == '__main__':
    unittest.main()
