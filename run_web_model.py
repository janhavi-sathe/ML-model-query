from web_model import create_app

app = create_app(debug=True)
app.config["DEBUG"] = True

if __name__ == '__main__':
    app.run(debug=True)