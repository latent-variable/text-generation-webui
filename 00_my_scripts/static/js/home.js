// Function to fetch JSON data and create the table
async function fetchAndDisplayJsonData() {
    try {
        // Fetch the JSON data from the server
        let response = await fetch('/get_json_data');
        let jsonData = await response.json();

        // Define the columns you want to display in the table
        let col = ['user_ip', 'feedback', 'timestamp'];

        // Create a table element
        const table = document.createElement("table");
        const thead = table.createTHead();
        const tbody = table.createTBody();

        table.setAttribute("id", "json-table");

        // Create the header row
        let tr = thead.insertRow(-1);
        for (let index = 0; index < col.length; index++) {
            let th = document.createElement("th");
            th.innerHTML = col[index];
            tr.appendChild(th);
        }

        // Create the rows for the table body
        for (let i = 0; i < jsonData.length; i++) {
            tr = tbody.insertRow(-1);
            for (let j = 0; j < col.length; j++) {
                let tabCell = tr.insertCell(-1);
                tabCell.innerHTML = jsonData[i][col[j]];
            }
        }

        // Append the table to the container
        document.querySelector(".json-table-container").appendChild(table);
    } catch (error) {
        console.error('Error fetching or displaying JSON data:', error);
    }
}

// Call the function to fetch and display the JSON data
fetchAndDisplayJsonData();