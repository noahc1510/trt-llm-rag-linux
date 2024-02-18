// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
() => {
    // file to customize behavior
    console.log('js loaded', window.location.href);

    //handle launch
    let reload = false;
    let gradioURL = new URL(window.location.href);
    if(gradioURL.searchParams.has('cookie')) {
        const secureCookie = gradioURL.searchParams.get('cookie');
        gradioURL.searchParams.delete('cookie');
        // to do set the cookie
        const SESSION_COOKIE_EXPIRY = 60 * 60 * 24; // 1 day
        //document.cookie = `_s_chat_=${secureCookie}; path=${location.pathname}; max-age=${SESSION_COOKIE_EXPIRY}; samesite=strict`;
        document.cookie = `_s_chat_=${secureCookie}; path=${location.pathname}; samesite=strict`;
        reload = true;
    }
    if(
        !gradioURL.searchParams.has('__theme') ||
        (gradioURL.searchParams.has('__theme') && gradioURL.searchParams.get('__theme') !== 'dark')
    ) {
        gradioURL.searchParams.delete('__theme');
        gradioURL.searchParams.set('__theme', 'dark');
        reload = true;
    }
    if(reload) {
        window.location.replace(gradioURL.href);
    }

    //tool tip
    var elements = document.getElementsByClassName("tooltip-component");
    console.log('elements found', elements.length)
    Array.prototype.forEach.call(elements, function (element) {
        // Get the tooltip content based on the element's ID
        var tooltipContent = getTooltipContent(element.id);
        console.log('tooltip content', tooltipContent);
        if(tooltipContent) {
            var tooltip = document.createElement("div");
            // var tooltipContentElement = document.getElementById("tooltip-content-element");
            // tooltipContentElement.forEach(className => {
            //     tooltip.classList.add(className);
            // });
            tooltip.classList.add("tooltip");
            tooltip.style.display = "none";
            tooltip.innerHTML = tooltipContent;
            document.body.appendChild(tooltip);
    
            element.addEventListener("mouseover", function () {
                console.log('mouse over', elements.length);
                var rect = element.getBoundingClientRect();
                tooltip.style.left = rect.left + "px";
                tooltip.style.top = (rect.bottom + 5) + "px";
                tooltip.style.display = "block";
            });
    
            element.addEventListener("mouseout", function () {
                tooltip.style.display = "none";
            });
        }
    });

    // Function to get tooltip content based on the element's ID
    function getTooltipContent(elementId) {
        // Example logic, you can customize it based on your needs
        if (elementId === "shutdown-btn") {
            return "Shutdown";
        } else if (elementId === "dataset-regenerate-index-btn") {
            return "Refresh dataset";
        } else if (elementId === "dataset-update-source-edit-button") {
                return "Change folder path";
        } else if (elementId === "dataset-update-source-download-button") {
                return "Download transcripts";
        } else if (elementId === "chatbot-retry-button") {
            return "Regenerate response";
        } else if (elementId === "chatbot-reset-button") {
            return "Clear chat";
        } else if (elementId === "chatbot-undo-button") {
            return "Undo";
        } else if (elementId === "dataset-view-transcript-folder-button") {
            return "View index folder";
        }

        return null;
    }

    // links in chatbot window
    let el = document.getElementById("main-chatbot-window");
    if(el) {
        el.addEventListener("click", (event) => {
            if(event.target.tagName.toLowerCase() === 'a' && event.target.href) {
                data = {
                    data: [event.target.href]
                }
                makeHttpCall('/api/open_file', 'POST', {}, data)
            }
        })
    } else {
        console.error("chatbot window not created");
    }

    function makeHttpCall(url, method, headers, data) {
        headers = Object.keys(headers).length !== 0 ? headers : {
            "Content-Type": "application/json"
        };
        options = {
            method,
            headers,
            body: JSON.stringify(data)
        };
        console.log(options);
        fetch(url, options)
            .then(response => response.json())
            .then(data => {
                console.log("Response:", data);
            })
            .catch(error => {
                console.error("Error:", error);
            });
    }
}