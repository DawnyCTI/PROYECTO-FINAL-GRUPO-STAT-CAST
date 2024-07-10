document.addEventListener("DOMContentLoaded", function () {
    var ctx = document.getElementById('confidenceChart').getContext('2d');
    var confidenceChart = new Chart(ctx, {
        type: 'bar', // Tipo de gráfico
        data: {
            labels: ['Confianza'], // Etiqueta para la barra
            datasets: [{
                label: 'Confianza (%)',
                data: [confidence], // Datos de confianza
                backgroundColor: ['rgba(75, 192, 192, 0.2)'],
                borderColor: ['rgba(75, 192, 192, 1)'],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100 // Máximo de la escala Y
                }
            },
            plugins: {
                legend: {
                    display: false // Ocultar la leyenda
                }
            }
        }
    });
});
