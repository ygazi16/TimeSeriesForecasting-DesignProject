//const form = document.getElementById('form');
//const username = document.getElementById('username');
//const email = document.getElementById('email');
//const password = document.getElementById('password');
//const repassword = document.getElementById('repassword');
const express = require('express');

const passport = require('passport');
const GoogleStrategy = require('passport-google-oauth20').Strategy;
const keys = require('/Users/yarkingazi/github/TimeSeriesForecasting-DesignProject/config/keys.js');

const app = express();

passport.use(new GoogleStrategy({
  clientID : keys.google.ClientID,
  clientSecret : keys.google.ClientSecret,
  callbackURL: 'auth/google/callback'
}, (accessToken) => {
  console.log(accessToken);
} ));

app.get('/auth/google',
passport.authenticate('google', {
  scope: ['profile', 'email']
})
);

//{"web":{"client_id":"511950486271-tuege6cfons5v671idgt2crhcgijvt52.apps.googleusercontent.com","project_id":"tsf-designproject","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_secret":"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","redirect_uris":["http://localhost:5000/auth/google/callback"],"javascript_origins":["http://127.0.0.1:3000"]}}

const PORT = process.env.PORT || 5000 || 3000;
app.listen(PORT);
