"""
Automated validation script for ImageLab backend APIs using modern ASGI transport.
"""

import asyncio
import cv2
import numpy as np
import httpx
from app.main import app


async def run_all_tests():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        print("================ RUNNING IMAGELAB BACKEND TESTS ================")
        
        # 1. Health check
        res = await client.get("/api/health")
        assert res.status_code == 200, f"Health check failed: {res.text}"
        data = res.json()
        assert data["status"] == "healthy"
        assert "ImageLab" in data["app"]
        print("✓ 1. Health check passed:", data)

        # 2. List all sample categories
        res = await client.get("/api/samples")
        assert res.status_code == 200, f"List all samples failed: {res.text}"
        data = res.json()
        assert data["status"] == "success"
        categories = data["categories"]
        print(f"✓ 2. Found {len(categories)} categories in dataset:")
        for cat, info in categories.items():
            print(f"     - {cat}: {info['count']} samples")
            assert info["count"] > 0, f"Category {cat} has 0 samples!"

        # 3. List and retrieve sample by category
        for cat in ["enhancement", "boundary", "watershed", "recognition"]:
            res = await client.get(f"/api/samples/{cat}")
            assert res.status_code == 200, f"List {cat} failed: {res.text}"
            data = res.json()
            assert data["count"] > 0
            first_file = data["samples"][0]
            
            img_res = await client.get(f"/api/samples/{cat}/{first_file}")
            assert img_res.status_code == 200, f"Get {cat}/{first_file} failed"
            assert len(img_res.content) > 0
            print(f"✓ 3. Category '{cat}' listed ({data['count']} files) & retrieved '{first_file}' ({len(img_res.content)} bytes)")

        # 4. Upload validation for PNG, JPG, BMP
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(test_img, (50, 50), 30, (0, 255, 0), -1)

        for fmt, ext in [("PNG", ".png"), ("JPEG", ".jpg"), ("BMP", ".bmp")]:
            _, buf = cv2.imencode(ext, test_img)
            files = {"file": (f"test_memory_upload{ext}", buf.tobytes(), f"image/{fmt.lower()}")}
            res = await client.post("/api/upload/validate", files=files)
            assert res.status_code == 200, f"Upload validation failed for {fmt}: {res.text}"
            val_data = res.json()
            assert val_data["status"] == "success"
            assert val_data["metadata"]["width"] == 100
            assert val_data["metadata"]["height"] == 100
            assert val_data["metadata"]["format"] == fmt.replace("JPEG", "JPG")
            print(f"✓ 4. In-memory upload accepted & verified for {fmt} (100x100 px, {val_data['metadata']['memory_size_bytes']} memory bytes)")

        # 5. Rejection tests
        # 5a. Invalid extension (.txt)
        files = {"file": ("script.txt", b"plain text payload", "text/plain")}
        res = await client.post("/api/upload/validate", files=files)
        assert res.status_code == 400
        assert "Unsupported file format" in res.json()["detail"]
        print("✓ 5a. Successfully rejected invalid file extension (.txt)")

        # 5b. Corrupted image bytes with valid extension
        files = {"file": ("damaged.png", b"GIF89a corrupt bytes not a valid png", "image/png")}
        res = await client.post("/api/upload/validate", files=files)
        assert res.status_code == 400
        assert "integrity check failed" in res.json()["detail"]
        print("✓ 5b. Successfully rejected corrupted image bytes")

        # 5c. Undersized image (8x8 < 16x16)
        tiny_img = np.zeros((8, 8), dtype=np.uint8)
        _, buf = cv2.imencode(".png", tiny_img)
        files = {"file": ("tiny.png", buf.tobytes(), "image/png")}
        res = await client.post("/api/upload/validate", files=files)
        assert res.status_code == 400
        assert "too small" in res.json()["detail"]
        print("✓ 5c. Successfully rejected undersized dimensions (8x8 < 16x16)")

        print("================ ALL BACKEND API TESTS PASSED SUCCESSFULLY ================")

if __name__ == "__main__":
    asyncio.run(run_all_tests())
