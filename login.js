const form = document.getElementById('form');
const username = document.getElementById('username');
const password = document.getElementById('password');


function error(input, message) {
  input.className = 'form-control is-invalid';
  const div = input.nextElementSibling;
  div.innerText = message;
  div.className = 'invalid-feedback';
}

function success(input) {
  input.className = 'form-control is-valid';
}



function checkRequired(inputs) {
  inputs.forEach(function(input) {
    if (input.value === '') {
      error(input, `${input.id} is required.`);
    } else {
      success(input);
    }
  });
}

function checkLength(input, min, max) {
  if (input.value.length < min) {
    error(input, `${input.id} should be at least ${min} characters`);
  } else if (input.value.length > max) {
    error(input, `${input.id} should be at most ${max} characters`);
  } else {
    success(input);
  }
}



form.addEventListener('submit', function(e) {
  e.preventDefault();

  checkRequired([username, password]);
  checkEmail(email);
  checkLength(username, 7, 15);
  checkLength(password, 7, 12);

});
