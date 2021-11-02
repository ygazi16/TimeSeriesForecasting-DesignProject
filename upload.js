  <input class="form-control form-control-lg" id="formFileLg" type="file" />
  const fs = require("fs");
  const mysql = require("mysql");
  const fastcsv = require("fast-csv");

  let stream = fs.createReadStream("formFileLg");
  let csvData = [];
  let csvStream = fastcsv
    .parse()
    .on("data", function(data) {
      csvData.push(data);
    })
    .on("end", function() {
      // remove the first line: header
      csvData.shift();

window.alert(csvData);
