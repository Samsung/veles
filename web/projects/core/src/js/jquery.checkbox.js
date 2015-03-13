(function ($) {
  $.widget("ui.checkbox", {
    _create: function() {
      this._super();
      this.element.addClass("ui-helper-hidden-accessible");
      this.button = $("<button/>").insertAfter(this.element);
      this.button.addClass("ui-checkbox").text("checkbox").button({
        text: !1,
        icons: {
          primary: "ui-icon-blank"
        },
        create: function () {
          $(this).removeAttr("title")
        }
      });
      this._on(this.button, {
        click: function () {
          this.element.prop("checked", !this.element.is(":checked"));
          this.refresh()
        }
      });
      this.refresh();
    },
    _destroy: function() {
      this._super();
      this.element.removeClass("ui-helper-hidden-accessible");
      this.button.button("destroy").remove();
    },
    refresh: function() {
      this.button.button("option", "icons", {
        primary: this.element.is(":checked") ? "ui-icon-check" : "ui-icon-blank"
      })
    }
  });

  $(function () {
    $("input[type='checkbox']").checkbox();
  });
})(jQuery);
