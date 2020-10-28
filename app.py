from flask import Flask, render_template, request
import pickle
import sklearn
from sklearn.preprocessing import PolynomialFeatures

dict_degree = pickle.load(open('dict_degree.pkl', 'rb'))
dict_model = pickle.load(open('dict_model.pkl', 'rb'))


app = Flask(__name__)

@app.route('/')
def man():
    return render_template('template.html')

@app.route('/predict', methods=['POST'])
def home():
    curb_weight = request.form['curb-weight']
    engine_size = request.form['engine-size']
    length = request.form['length']
    width = request.form['width']
    horsepower = request.form['horsepower']
    city_mpg = request.form['city-mpg']
    highway_mpg = request.form['highway-mpg']
    wheel_base = request.form['wheel-base']
    bore = request.form['bore']

    dict_ft = {'curb-weight': curb_weight,
               'engine-size': engine_size,
               'length': length,
               'width': width,
               'horsepower': horsepower,
               'city-mpg': city_mpg,
               'highway-mpg': highway_mpg,
               'wheel-base': wheel_base,
               'bore': bore
               }

    # lấy các feature có giá trị khác None
    list_ft = []
    list_vl = []
    for key, value in dict_ft.items():
        if value != '':
            list_ft.append(key)
            list_vl.append(value)

    if list_ft == []:
        return render_template('template2.html', data='PLEASE ENTER A VALUE!')

    # Polynomial X
    X = [list_vl]
    pf = PolynomialFeatures(dict_degree[str(list_ft)], include_bias=False)
    X_poly = pf.fit_transform(X)

    # lấy model đã được huấn luyện ở câu trước
    model = dict_model[str(list_ft)]
    y_pred = model.predict(X_poly)

    y_pred = str(y_pred).lstrip('[[').rstrip(']]')

    return render_template('template2.html', data=y_pred)

if __name__ == "__main__":
    app.run(debug=True)
