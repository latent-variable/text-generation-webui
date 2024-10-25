JS_FUNC = """
    function refresh() {
        const url = new URL(window.location);

        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
    """
    
CSS_FORMAT="""
    .header-row { 
        display: flex; 
        justify-content: space-between; 
        align-items: flex-start; /* Align items to the top */
    }
    .title-disclaimer-container {
        display: block; /* Allows the disclaimer to be below the title */
        
    }
    .title { 
        font-size: 48px; 
        font-weight: bold; 
    } 
    .disclaimer { 
        color: #aaa; 
        margin-top: 20px; /* Increased space below the title */
    }
    .logo-container {
        /* Ensures that the logo sticks to the right and is aligned to the top */
        margin-right: auto; /* 500px Add some space between the logo and the title */
        display: flex; 
        align-items: flex-end;
        float: right;
    }
    .logo {
        height: 100px; /* Adjust the height as needed */
        width: auto; /* Ensure the width adjusts automatically */
    }
    .bottom-disclaimer {
        text-align: center;
        display:block;
    }
    footer {visibility: hidden}
    """