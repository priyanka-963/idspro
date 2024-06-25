document.addEventListener('DOMContentLoaded', function () {
    var chartContainer = document.querySelector(".chart-container");

    async function fetchInterfaces() {
        try {
            const response = await fetch('/get-interfaces');
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            const data = await response.json();
            return data.interfaces;
        } catch (error) {
            console.error('Error fetching interfaces:', error);
            return [];
        }
    }

    function updateClock() {
        const now = new Date();
        const hours = now.getHours();
        const minutes = now.getMinutes();
        const seconds = now.getSeconds();
        const ampm = hours >= 12 ? 'PM' : 'AM';
        const hours12 = hours % 12 || 12;
        const timeString = `${hours12}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')} ${ampm}`;
        const dateString = now.toDateString();
        document.getElementById('time').textContent = timeString;
        document.getElementById('date').textContent = dateString;
    }

    async function fetchLocation() {
        try {
            const response = await fetch('https://ipapi.co/json/');
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            const data = await response.json();
            document.getElementById('location').textContent = data.city + ", " + data.region + ", " + data.country;
        } catch (error) {
            console.error('Error fetching location:', error);
        }
    }

    async function fetchTrafficData() {
        try {
            const response = await fetch('/firewalltraffic');
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching traffic data:', error);
            return {};
        }
    }

    async function initialize() {
        const interfaces = await fetchInterfaces();
        const chartData = interfaces.map(interface => ({ name: interface, data: [] }));
        
        const comparisonChart = new ApexCharts(document.querySelector("#comparison-chart"), {
            chart: {
                type: 'line',
                height: '100%',
                animations: {
                    enabled: true,
                    easing: 'linear',
                    dynamicAnimation: {
                        speed: 1000
                    }
                }
            },
            series: chartData,
            xaxis: {
                type: 'datetime'
            }
        });

        comparisonChart.render();

        setInterval(updateClock, 1000);
        fetchLocation();

        async function updateData() {
            const trafficData = await fetchTrafficData();
            const now = new Date().getTime();

            chartData.forEach((interfaceData, interfaceIndex) => {
                const trafficValue = trafficData[interfaceData.name];
                if (trafficValue !== undefined) {
                    interfaceData.data.push({ x: now, y: trafficValue });
                    if (interfaceData.data.length > 20) {
                        interfaceData.data.shift();
                    }
                }
            });

            comparisonChart.updateSeries(chartData);
        }

        setInterval(updateData, 1000);
    }

    initialize();
});
