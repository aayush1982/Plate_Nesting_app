# 🧩 Nesting App

**Nesting App** is a tool developed to optimize plate cutting and nesting for connection parts and built-up beam/column (BH/CT) profiles.

---

## 🚀 Features
- Automatic nesting of connection and plate components  
- Intelligent plate utilization with tail merge logic  
- PDF report generation including:
  - Plate summary with usage category (BH/CT, BH/CT + Conn, Conn)
  - Splice view diagrams
  - Nesting criteria and assumptions
- Supports modular enhancement (e.g., DXF-based input for future)

---

## 🧮 Nesting Criteria
- Max one splice joint per member  
- Joints are positioned in the **middle third** of height  
- Web and flange splices are **staggered**  
- Minimum **300 mm clearance** between web splice and flange weld  
- Plate widths up to **3500 mm** (for ≤45 mm thick plates)  
- Leftover plates reused for connection nesting  

---


