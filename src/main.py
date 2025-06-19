#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face Attendance System - GUI Main
Hệ thống điểm danh bằng nhận dạng khuôn mặt - Giao diện đồ họa
"""

import sys
import os

# Thêm thư mục hiện tại vào path để import được các module  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui import main

if __name__ == "__main__":
    main()