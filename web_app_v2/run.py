from web_experiment import create_app, socketio

app = create_app(debug=True)

if __name__ == '__main__':
  socketio.run(app, debug=True, use_reloader=True)
