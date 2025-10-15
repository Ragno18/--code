document.getElementById('analyse-btn').addEventListener('click', () => {
    const minPrice = document.getElementById('min_price').value;
    const maxPrice = document.getElementById('max_price').value;

    const queryParams = new URLSearchParams({
        min_price: minPrice,
        max_price: maxPrice,
    });

    fetch(`http://127.0.0.1:5000/api/stocks?${queryParams}`)
        .then(response => response.json())
        .then(stocks => {
            const tableBody = document.getElementById('stock-table-body');
            tableBody.innerHTML = ''; // Clear existing data
            stocks.forEach(stock => {
                const row = `<tr>
                    <td>${stock.symbol}</td>
                    <td>${stock.price.toFixed(2)}</td>
                </tr>`;
                tableBody.innerHTML += row;
            });
        })
        .catch(error => {
            console.error('Error fetching stock data:', error);
            const tableBody = document.getElementById('stock-table-body');
            tableBody.innerHTML = '<tr><td colspan="2">Error loading data.</td></tr>';
        });
});