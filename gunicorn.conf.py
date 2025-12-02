workers = 1               # Only 1 worker for TensorFlow (hemat RAM)
threads = 2               # Thread kecil cukup
bind = "0.0.0.0:8004"
timeout = 180
preload_app = True        # Load TensorFlow model sekali saja
