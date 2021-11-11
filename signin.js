const form = document.getElementById('form');
const username = document.getElementById('username');
const email = document.getElementById('email');
const password = document.getElementById('password');
const repassword = document.getElementById('repassword');
const express = require('express');

const passport = require('passport');
const GoogleStrategy = require('passport-google-oauth20').Strategy;
const keys = require('./config/keys.');

const app = express();

passport.use(new GoogleStrategy({
  clientID = keys.google.ClientID,
  clientSecret = keys.google.ClientSecret,
  callbackURL: 'auth/google/callback'
}, (accessToken) => {
  console.log(accessToken);
} ));


//{"web":{"client_id":"511950486271-tuege6cfons5v671idgt2crhcgijvt52.apps.googleusercontent.com","project_id":"tsf-designproject","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_secret":"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","redirect_uris":["http://localhost:5000/auth/google/callback"],"javascript_origins":["http://127.0.0.1:3000"]}}

const PORT = process.env.PORT || 5000;
app.listen(PORT);






function error(input, message) {
    input.className = 'form-control is-invalid';
    const div = input.nextElementSibling;
    div.innerText = message;
    div.className = 'invalid-feedback';
}

function success(input) {
    input.className = 'form-control is-valid';
}

function checkEmail(input) {
    const re = /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;

    if(re.test(input.value)) {
        success(input);
    } else {
        error(input, 'wrong email format!');
    }
}

function checkRequired(inputs) {
    inputs.forEach(function(input) {
        if(input.value === '') {
            error(input, `${input.id} is required.`);
        } else {
            success(input);
        }
    });
}

function checkLength(input, min, max) {
    if (input.value.length < min) {
        error(input, `${input.id} should be at least ${min} characters`);
    }else if (input.value.length > max) {
        error(input, `${input.id} should be at most ${max} characters`);
    }else {
        success(input);
    }
}

function checkPasswords(input1,input2) {
    if(input1.value !== input2.value) {
        error(input2, 'passwords do not match!');
    }
}

function checkPhone(input) {
    var exp = /^\d{10}$/;
    if(!exp.test(input.value))
        error(input, 'phone should have at least 10 characters');
}

form.addEventListener('submit', function(e) {
    e.preventDefault();

    checkRequired([username,email,password,repassword,phone]);
    checkEmail(email);
    checkLength(username,7,15);
    checkLength(password,7,12);
    checkPasswords(password,repassword);
    checkPhone(phone);
});
