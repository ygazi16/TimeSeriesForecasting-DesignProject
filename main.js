const path = "file:///C:/Users/ASUS/OneDrive/Desktop/upload/upload.html";
const searchBox = document.querySelector(".search-box");
const navBtnContainer = document.querySelector(".nav-btn-container");
const searchBtn = document.querySelector(".search-btn");
const closeBtn = document.querySelector(".close-btn");

const models = document.querySelector("#models");
const upload = document.querySelector("#upload");
const signin = document.querySelector("#signin");

upload.addEventListener("click", () => {
  window.location.href = path;
});

signin.addEventListener("click", () => {
  window.location.href = path;
});

searchBtn.addEventListener("click", () => {
  searchBox.classList.add("active");
  navBtnContainer.classList.add("active");
});

closeBtn.addEventListener("click", () => {
  searchBox.classList.remove("active");
  navBtnContainer.classList.remove("active");
});
