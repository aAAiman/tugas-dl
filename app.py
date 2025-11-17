from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from model_bidirectional import run_bidirectional_model
from model_lstm import predict_stock
from model_image import classify_image
import os

app = Flask(__name__)

# folder upload untuk tugas 4
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


# ================== TUGAS 1: Kalkulator Operator Logika ==================
@app.route("/tugas1", methods=["GET", "POST"])
def tugas1():
    result = None
    a = ''
    b = ''
    op = ''

    if request.method == "POST":
        a = request.form["a"]
        b = request.form["b"]
        op = request.form["operator"].upper()

        # Validasi input hanya 0/1
        if not all(ch in "01" for ch in a):
            result = "Input A harus berupa biner (0/1)"
            return render_template("tugas1.html", result=result, a=a, b=b, op=op)

        if op != "NOT" and not all(ch in "01" for ch in b):
            result = "Input B harus berupa biner (0/1)"
            return render_template("tugas1.html", result=result, a=a, b=b, op=op)

        # Convert ke integer
        a_int = int(a, 2)

        if op != "NOT":
            b_int = int(b, 2)

        # Operasi bitwise
        if op == "AND":
            result_int = a_int & b_int
        elif op == "OR":
            result_int = a_int | b_int
        elif op == "XOR":
            result_int = a_int ^ b_int
        elif op == "NOT":
            # NOT: hasil harus disesuaikan panjang bit
            length = len(a)
            result_int = (~a_int) & ((1 << length) - 1)
        else:
            result = "Operator tidak dikenal"
            return render_template("tugas1.html", result=result, a=a, b=b, op=op)

        # Convert hasil ke biner
        result = bin(result_int)[2:]

    return render_template("tugas1.html", result=result, a=a, b=b, op=op)



# ================== TUGAS 2: Prediksi Kata (Bidirectional) ==================
@app.route("/tugas2", methods=["GET", "POST"])
def tugas2():
    input_text = ""
    prediction = None

    if request.method == "POST":
        input_text = request.form.get("input_text", "").strip()

        if input_text:
            prediction = run_bidirectional_model(input_text)
        else:
            prediction = "Input kosong."

    return render_template("tugas2.html",
                           input_text=input_text,
                           prediction=prediction)

# ================== TUGAS 3: Prediksi Harga Saham ==================
from model_lstm import predict_stock

@app.route("/tugas3", methods=["GET", "POST"])
def tugas3():
    stock_code = ""
    n_days = ""
    prediction_result = None

    if request.method == "POST":
        stock_code = request.form.get("stock_code", "").upper()
        n_days = request.form.get("n_days", "1")

        try:
            n_days_int = int(n_days)
        except ValueError:
            n_days_int = 1

        hasil = predict_stock(stock_code, n_days_int)
        prediction_result = hasil

    return render_template("tugas3.html",
                           stock_code=stock_code,
                           n_days=n_days,
                           prediction_result=prediction_result)

# ================== TUGAS 4: Mengenal Object (Image Classification) ==================
@app.route("/tugas4", methods=["GET", "POST"])
def tugas4():
    filename = None
    predicted_label = None

    if request.method == "POST":
        if "image" not in request.files:
            predicted_label = "Tidak ada file yang dikirim."
        else:
            file = request.files["image"]
            if file.filename == "":
                predicted_label = "File belum dipilih."
            else:
                safe_name = secure_filename(file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
                file.save(save_path)

                filename = safe_name

                # klasifikasi beneran
                predicted_label = classify_image(save_path)

    return render_template("tugas4.html",
                           filename=filename,
                           predicted_label=predicted_label)




if __name__ == "__main__":
    app.run(debug=True)
