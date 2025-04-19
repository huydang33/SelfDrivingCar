## Self Driving Car
kết hợp Lane Segmentation và Traffic Sign Detection
🧩 Option 1: Hai model độc lập (Lane + Sign detection)
❖ Mô tả:
- Dùng một model cho lane segmentation (UNet, ResNet-based),
- Dùng model riêng cho object detection (YOLOv8, Faster R-CNN...) để detect traffic signs.

✅ Ưu điểm:
- Dễ huấn luyện, dễ debug.
- Reuse model SOTA sẵn có (YOLOv5/v8 cho traffic signs).
- Độc lập dễ maintain, dễ scale.

❌ Nhược điểm:
- Tốn tài nguyên (2 lần inference).
- Khó tối ưu end-to-end performance.

▶️ Phù hợp nếu:
- Build một hệ thống modular.
- Cần flexibility để thay thế 1 model mà không ảnh hưởng toàn hệ thống.

🧠 Option 2: Multi-task Learning Model (1 model 2 task)
❖ Mô tả:
- Một encoder dùng chung (ResNet18, ResNet50…),

Hai decoder:
1. Lane decoder → segmentation.
2. Sign decoder → detection (bounding boxes hoặc classification head).

✅ Ưu điểm:
- Tiết kiệm tài nguyên hơn (share encoder).
- Có thể học representation chung tốt hơn nếu hai task liên quan.

❌ Nhược điểm:
- Cần dataset chứa cả lane + sign annotation (khó tìm).
- Huấn luyện phức tạp hơn (multi-loss balancing).
- Có thể gây conflict giữa hai task nếu không tương thích.

▶️ Phù hợp nếu:
- Tối ưu hiệu năng (real-time inference).
- Có dataset chất lượng hoặc khả năng tạo dataset dạng multi-task.

🧪 Option 3: Pipeline 2 bước (Lane trước → Region crop → detect sign)
❖ Mô tả:
Dùng segmentation để biết đường đi.
Chỉ detect sign ở vùng gần đường, hoặc vùng tập trung (ROI).
Giảm search space cho detector.

✅ Ưu điểm:
- Tăng tốc độ inference cho detector.
- Khả năng tăng accuracy bằng cách focus vùng liên quan.

❌ Nhược điểm:
- Phức tạp khi implement pipeline xử lý.
- Dễ lỗi nếu bước đầu segmentation không tốt.