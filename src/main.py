from pytermgui.enums import CenteringPolicy, HorizontalAlignment
from pytermgui import keys
import pytermgui as ptg

file = open('test.txt', 'w')

with ptg.WindowManager() as manager:
    title_style = ptg.MarkupFormatter("[lightgreen]{item}[/]")
    main = ptg.Container(
        ptg.Splitter(ptg.Label("->", parent_align=2),
                     ptg.Label("Test NER", parent_align=0)), box="EMPTY")

    def replace(widget, key):
        main[0] = ptg.Splitter(ptg.Label("Sentence"),
                               ptg.InputField(multiline=True))
        main[0].select(index=1)

    main[0].bind(keys.RETURN, replace)

    window = (
        ptg.Window(
            ptg.Container("[lightgreen]News Recommender[/]",
                          box="DOUBLE", relative_width=0.5)
            .set_style("value", title_style)
            .set_style("border", title_style)
            .set_style("corner", title_style)
            .center(where=CenteringPolicy.HORIZONTAL),
            "Welcome to our News Recommender System!",
            main,
            box="EMPTY"
        )
        .center()
    )
    manager.add(window)


file.close()
