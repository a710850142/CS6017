// Set up dimensions and margins
const margin = {top: 50, right: 30, bottom: 70, left: 60};
const width = 500 - margin.left - margin.right;
const height = 400 - margin.top - margin.bottom;

// Create SVG for bar chart
const svgBar = d3.select("#barChart")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

// Create SVG for scatter plot
const svgScatter = d3.select("#scatterPlot")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

// Create tooltip
const tooltip = d3.select("body").append("div")
    .attr("class", "tooltip")
    .style("opacity", 0);

// Load data
d3.csv("Car_sales.csv").then(function(data) {
    // Data preprocessing
    data.forEach(d => {
        d.Sales_in_thousands = +d.Sales_in_thousands;
        d.Price_in_thousands = +d.Price_in_thousands;
        d.Fuel_efficiency = +d.Fuel_efficiency;
    });

    // Sort by sales and get top 15 manufacturers
    const topManufacturers = data.sort((a, b) => b.Sales_in_thousands - a.Sales_in_thousands)
        .slice(0, 15)
        .map(d => d.Manufacturer);

    // Filter data
    const filteredData = data.filter(d => topManufacturers.includes(d.Manufacturer));

    // Create bar chart
    createBarChart(filteredData);

    // Create scatter plot
    createScatterPlot(filteredData);
});

function createBarChart(data) {
    // Set up x-axis
    const x = d3.scaleBand()
        .range([0, width])
        .domain(data.map(d => d.Manufacturer))
        .padding(0.1);

    svgBar.append("g")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(x))
        .selectAll("text")
        .attr("transform", "translate(-10,0)rotate(-45)")
        .style("text-anchor", "end");

    // Set up y-axis
    const y = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.Sales_in_thousands)])
        .range([height, 0]);

    svgBar.append("g")
        .call(d3.axisLeft(y));

    // Create bars
    svgBar.selectAll("bars")
        .data(data)
        .enter()
        .append("rect")
        .attr("x", d => x(d.Manufacturer))
        .attr("y", d => y(d.Sales_in_thousands))
        .attr("width", x.bandwidth())
        .attr("height", d => height - y(d.Sales_in_thousands))
        .attr("fill", "#69b3a2")
        .on("mouseover", function(event, d) {
            d3.select(this).attr("fill", "orange");
            tooltip.transition()
                .duration(200)
                .style("opacity", .9);
            tooltip.html(`Brand: ${d.Manufacturer}<br/>Sales: ${d.Sales_in_thousands}k`)
                .style("left", (event.pageX) + "px")
                .style("top", (event.pageY - 28) + "px");
            highlightScatter(d.Manufacturer);
        })
        .on("mouseout", function(d) {
            d3.select(this).attr("fill", "#69b3a2");
            tooltip.transition()
                .duration(500)
                .style("opacity", 0);
            resetScatter();
        });

    // Add title
    svgBar.append("text")
        .attr("x", width / 2)
        .attr("y", 0 - margin.top / 2)
        .attr("text-anchor", "middle")
        .style("font-size", "16px")
        .text("Car Brand Sales (thousands)");
}

function createScatterPlot(data) {
    // Set up x-axis
    const x = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.Price_in_thousands)])
        .range([0, width]);

    svgScatter.append("g")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(x));

    // Set up y-axis
    const y = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.Fuel_efficiency)])
        .range([height, 0]);

    svgScatter.append("g")
        .call(d3.axisLeft(y));

    // Create dots
    svgScatter.selectAll("dot")
        .data(data)
        .enter()
        .append("circle")
        .attr("cx", d => x(d.Price_in_thousands))
        .attr("cy", d => y(d.Fuel_efficiency))
        .attr("r", 5)
        .attr("fill", "#69b3a2")
        .attr("class", d => d.Manufacturer)
        .on("mouseover", function(event, d) {
            d3.select(this).attr("r", 8).attr("fill", "orange");
            tooltip.transition()
                .duration(200)
                .style("opacity", .9);
            tooltip.html(`Brand: ${d.Manufacturer}<br/>Price: $${d.Price_in_thousands}k<br/>Fuel Efficiency: ${d.Fuel_efficiency} MPG`)
                .style("left", (event.pageX) + "px")
                .style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", function(d) {
            d3.select(this).attr("r", 5).attr("fill", "#69b3a2");
            tooltip.transition()
                .duration(500)
                .style("opacity", 0);
        });

    // Add title
    svgScatter.append("text")
        .attr("x", width / 2)
        .attr("y", 0 - margin.top / 2)
        .attr("text-anchor", "middle")
        .style("font-size", "16px")
        .text("Price vs Fuel Efficiency");

    // Add x-axis label
    svgScatter.append("text")
        .attr("x", width / 2)
        .attr("y", height + margin.bottom - 10)
        .style("text-anchor", "middle")
        .text("Price (thousands $)");

    // Add y-axis label
    svgScatter.append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", 0 - height / 2)
        .attr("y", 0 - margin.left)
        .attr("dy", "1em")
        .style("text-anchor", "middle")
        .text("Fuel Efficiency (MPG)");

    // Add zoom functionality
    const zoom = d3.zoom()
        .scaleExtent([1, 8])
        .on("zoom", zoomed);

    svgScatter.call(zoom);

    function zoomed(event) {
        svgScatter.selectAll("circle")
            .attr("transform", event.transform);
        svgScatter.select(".x-axis").call(x.scale(event.transform.rescaleX(x)));
        svgScatter.select(".y-axis").call(y.scale(event.transform.rescaleY(y)));
    }
}

function highlightScatter(manufacturer) {
    svgScatter.selectAll("circle")
        .attr("opacity", 0.2)
        .filter(`.${manufacturer}`)
        .attr("opacity", 1)
        .attr("r", 8)
        .attr("fill", "orange");
}

function resetScatter() {
    svgScatter.selectAll("circle")
        .attr("opacity", 1)
        .attr("r", 5)
        .attr("fill", "#69b3a2");
}