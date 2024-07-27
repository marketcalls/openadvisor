document.addEventListener('DOMContentLoaded', () => {
    const themeController = document.querySelectorAll('.theme-toggle-button');
    const themeToggleContainer = document.querySelector('.theme-toggle-container');

    const updateTheme = (theme) => {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);

        // Update theme-toggle-container background color based on the theme
        if (theme === 'dark') {
            themeToggleContainer.style.backgroundColor = '#333';
        } else if (theme === 'night') {
            themeToggleContainer.style.backgroundColor = '#1a1a2e';
        } else if (theme === 'light') {
            themeToggleContainer.style.backgroundColor = '#0f4c81';
        }
    };

    // Load the stored theme from local storage
    const storedTheme = localStorage.getItem('theme');
    if (storedTheme) {
        updateTheme(storedTheme);
    }

    themeController.forEach(button => {
        button.addEventListener('click', (event) => {
            const theme = event.currentTarget.getAttribute('data-theme');
            updateTheme(theme);
        });
    });
});
