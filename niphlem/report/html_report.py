import weakref
from html import escape
import warnings

MAX_IMG_VIEWS_BEFORE_WARNING = 10


class HTMLDocument(object):
    """
    Embeds a plot in a web page.
    If you are running a Jupyter notebook, the plot will be displayed
    inline if this object is the output of a cell.
    Otherwise, use open_in_browser() to open it in a web browser (or
    save_as_html("filename.html") to save it as an html file).
    use str(document) or document.html to get the content of the web page,
    and document.get_iframe() to have it wrapped in an iframe.
    """
    _all_open_html_repr = weakref.WeakSet()

    def __init__(self, html, width=600, height=400):
        self.html = html
        self.width = width
        self.height = height
        self._temp_file = None
        self._check_n_open()

    def _check_n_open(self):
        HTMLDocument._all_open_html_repr.add(self)
        if MAX_IMG_VIEWS_BEFORE_WARNING is None:
            return
        if MAX_IMG_VIEWS_BEFORE_WARNING < 0:
            return
        if len(HTMLDocument._all_open_html_repr
               ) > MAX_IMG_VIEWS_BEFORE_WARNING - 1:
            warnings.warn('It seems you have created more than {} '
                          'nilearn views. As each view uses dozens '
                          'of megabytes of RAM, you might want to '
                          'delete some of them.'.format(
                              MAX_IMG_VIEWS_BEFORE_WARNING))

    def resize(self, width, height):
        """Resize the plot displayed in a Jupyter notebook."""
        self.width, self.height = width, height
        return self

    def get_iframe(self, width=None, height=None):
        """
        Get the document wrapped in an inline frame.
        For inserting in another HTML page of for display in a Jupyter
        notebook.
        """
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        escaped = escape(self.html, quote=True)
        wrapped = ('<iframe srcdoc="{}" width="{}" height="{}" '
                   'frameBorder="0"></iframe>').format(escaped, width, height)
        return wrapped

    def get_standalone(self):
        """ Get the plot in an HTML page."""
        return self.html

    def _repr_html_(self):
        """
        Used by the Jupyter notebook.
        Users normally won't call this method explicitly.
        """
        return self.get_iframe()

    def __str__(self):
        return self.html

    def save_as_html(self, file_name):
        """
        Save the plot in an HTML file, that can later be opened in a browser.
        """
        with open(file_name, 'wb') as f:
            f.write(self.get_standalone().encode('utf-8'))


class HTMLReport(HTMLDocument):
    """A report written as HTML.
    Methods such as save_as_html(), open_in_browser()
    are inherited from HTMLDocument
    """

    def __init__(self, head_tpl, body, head_values={}):
        """The head_tpl is meant for display as a full page, eg writing on
        disk. The body is used for embedding in an existing page.
        """
        html = head_tpl.safe_substitute(body=body, **head_values)
        super(HTMLReport, self).__init__(html)
        self.head_tpl = head_tpl
        self.body = body

    def _repr_html_(self):
        """
        Used by the Jupyter notebook.
        Users normally won't call this method explicitly.
        """
        return self.body

    def __str__(self):
        return self.body
