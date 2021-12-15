from app.models import User,Graph
import os
from app import app, db
from flask import render_template, url_for, flash, redirect, request
from app.forms import LoginForm, RegistrationForm, UploadForm
from flask_login import login_user, current_user, logout_user, login_required
import csv

@app.route("/home")
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html", title="About")

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
         user = User(name=form.name.data, surname= form.surname.data, username=form.username.data, email=form.email.data, birthyear=form.birthyear.data, password=form.password.data, area=form.area.data)
         db.session.add(user)
         db.session.commit()
         flash(f'Your account created successfully', 'success')       
         return redirect(url_for('login'))
    return render_template('signin.html', title='Sign Up', form=form)

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and (user.password == form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))

def save_file(form_file):
    random_name= "rsndom_file_name"
    f_name, f_ext = os.path.splitext(form_file.filename)
    f_total = f_name + f_ext
    file_path = os.path.join(app.root_path, 'static/uploaded', f_total)
    form_file.save(file_path)
    return f_total

@app.route("/profile", methods=['GET', 'POST'])
@login_required
def profile():
    form = UploadForm()
    if form.validate_on_submit():
        if form.file.data:
           uploaded_file = save_file(form.file.data)
           current_user.csv_file = uploaded_file      
           db.session.commit()
           flash('Your file has been loaded!', 'success')
        else:
            flash('You did not upload any file!', 'danger')
        return redirect(url_for('profile'))    
    return render_template('profile.html', title='Profile',
                           form=form)

   # profile_pic = url_for('static', filename='profile picture/anon.jpg')
   # return render_template('profile.html', title='My Profile')

from app.graph import deneme, logicalInformationGraph


@app.route("/graph",methods=['GET', 'POST'])
def graph():  
    data = deneme()
    info= logicalInformationGraph()
    information = []
    information.append(info)

    
    legend = "data for " + current_user.csv_file
    labels = []
    values = []

    
    print(current_user.csv_file)
    for row in data:
        labels.append(row[0])
        values.append(row[1])
    return render_template('graph.html', title="My Graphs", values=values, labels=labels, legend=legend, information=information)




    








