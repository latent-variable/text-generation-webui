async function fetchAndDisplayJsonData(timeframe) {
    try {
        // Fetch the JSON data from the server based on selected timeframe
        let response = await fetch(`/get_json_data?timeframe=${timeframe}`);
        let jsonData = await response.json();

        // Define the columns you want to display in the table
        let col = ['user_ip', 'feedback', 'timestamp', 'addressed'];

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
                if (col[j] === 'addressed') {
                    // Create a checkbox element for 'addressed' field
                    let checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.checked = jsonData[i][col[j]]; // Set initial state based on JSON data
                    checkbox.addEventListener('change', function() {
                        updateAddressedStatus(jsonData[i], this.checked);
                    });
                    tabCell.appendChild(checkbox);
                } else {
                    tabCell.innerHTML = jsonData[i][col[j]];
                }
            }
        }

        // Clear previous table content
        document.querySelector(".json-table-container").innerHTML = '';

        // Append the table to the container
        document.querySelector(".json-table-container").appendChild(table);
    } catch (error) {
        console.error('Error fetching or displaying JSON data:', error);
    }
}

// Function to update 'addressed' status and save to server
async function updateAddressedStatus(rowData, checked) {
    try {
        rowData['addressed'] = checked;
        // Update the JSON data on the server using fetch and POST method
        await fetch('/update_addressed_status', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(rowData)
        });
    } catch (error) {
        console.error('Error updating addressed status:', error);
    }
}

// Function to handle timeframe change and fetch data
function handleTimeframeChange() {
    let timeframe = document.querySelector('select[name="timeframe"]').value;
    fetchAndDisplayJsonData(timeframe);
}

// Event listener for timeframe change
document.querySelector('select[name="timeframe"]').addEventListener('change', handleTimeframeChange);

// Fetch and display initial data on page load
document.addEventListener('DOMContentLoaded', function() {
    let initialTimeframe = document.querySelector('select[name="timeframe"]').value;
    fetchAndDisplayJsonData(initialTimeframe);
});