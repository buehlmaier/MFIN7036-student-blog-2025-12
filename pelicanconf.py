AUTHOR = 'MFIN7036 Students 2025'
SITENAME = 'MFIN7036 Student Blog 2025'
SITEURL = ''

PATH = 'content'

TIMEZONE = 'Asia/Hong_Kong'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (('Pelican', 'https://getpelican.com/'),
         ('Python.org', 'https://www.python.org/'),
         ('Jinja2', 'https://palletsprojects.com/p/jinja/'),
         ('You can modify those links in your config file', '#'),)

# Social widget
SOCIAL = (
    ('Github', 'https://github.com/buehlmaier/MFIN7036-student-blog-2025-12'),
    ('You can add links in your config file', '#'),
    ('Another social link', '#'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

# For "Elegant" theme landing page:
LANDING_PAGE_TITLE = "HKU MFIN7036 Student Blog 2025-2026"

DIRECT_TEMPLATES = ('index', 'archives', 'categories', 'tags', 'authors', 'search')
LUNR_LANG = 'en'
MARKDOWN = {
    'extensions': ['extra',
                   # 'mdx_math',
                   'codehilite'],
    'extension_configs': {
        # 'mdx_math': {'enable_dollar_delimiter': True},
        'codehilite': {'guess_lang': False},
    },
    'output_format': 'html5',
}
PLUGINS = ['render_math']
MATH_JAX = {
    'auto_insert': True, 
    'tex_extensions': ['color.js', 'mhchem.js'],
}
